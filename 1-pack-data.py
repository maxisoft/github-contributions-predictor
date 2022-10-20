import datetime
import json
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

user_columns = ['age', 'followers', 'following', 'gistComments', 'gists', 'issueComments', 'issues', 'organizations',
                'projectsV2', 'pullRequests', 'repositories', 'repositoriesContributedTo', 'watching',
                'pullRequestContributions', 'issueContributions', 'totalContributions']


def pack_data(out='userdata', align_weekday=True):
    current_year = datetime.date.today().year
    with open('contribs.json') as f:
        contribs: dict[str, dict] = json.load(f)

        user_buffer = np.zeros((len(contribs), len(user_columns)), dtype=np.float32)
        user_indices = np.zeros(len(contribs), dtype='<U32')
        c = 0
        contributions_buffer = np.zeros(shape=(len(contribs), 7 * 32), dtype=np.int32)
        for user_index, (user, data) in enumerate(contribs.items()):
            user_indices[c] = user
            try:
                user_data: dict = data['data']['user']
            except KeyError:
                log.warning("unable to get user data", exc_info=True)
                continue

            def ci(column: str) -> int:
                return user_columns.index(column)

            try:
                user_buffer[c, ci('age')] = current_year - min(
                    user_data['contributionsCollection']['contributionYears'])
                contributions: dict = user_data['contributionsCollection']
            except (TypeError, KeyError):
                continue

            user_buffer[c, ci('totalContributions')] = contributions['contributionCalendar']['totalContributions']

            for col_index, col in enumerate(user_columns):
                if col in user_data:
                    val = user_data.get(col)
                    if isinstance(val, (int, float)):
                        user_buffer[c, col_index] = val
                    elif "totalCount" in val:
                        user_buffer[c, col_index] = val['totalCount']
                if col in contributions:
                    val = contributions.get(col)
                    if "totalCount" in val:
                        user_buffer[c, col_index] = val['totalCount']

            contribution_counter = 0
            weeks = contributions['contributionCalendar']['weeks']
            if len(weeks) < contributions_buffer.shape[1] // 7:
                log.warning("%s doesn't have enough contribution %d", user, len(weeks))
                continue

            for week in weeks[-contributions_buffer.shape[1] // 7:]:
                for weekday, day in enumerate(week['contributionDays']):
                    assert weekday == day['weekday']
                    contributions_buffer[c, contribution_counter] = min(max(day['contributionCount'], 0),
                                                                                  2 ** 16)
                    contribution_counter += 1
                if not align_weekday and weekday != 6:
                    break

            if contribution_counter < contributions_buffer.shape[1]:
                shift = (contributions_buffer.shape[1] - contribution_counter)
                if align_weekday:
                    shift //= 7
                    shift *= 7
                contributions_buffer[c, :] = np.roll(contributions_buffer[c, :], shift)

            if contribution_counter > contributions_buffer.shape[1] // 2:
                c += 1

        user_df = pd.DataFrame(user_buffer[:c, :], index=user_indices[:c], columns=user_columns)
        #print(user_df)
        np.savez_compressed(out, users=user_buffer[:c, :], indices=user_indices[:c],
                            weeks=contributions_buffer[:c, :], user_columns=np.asarray(user_columns))
        return user_df


if __name__ == '__main__':
    pack_data()
