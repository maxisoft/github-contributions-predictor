import asyncio
import getpass
import hashlib
import json
import logging
import os
import pathlib
import random
import time
import warnings

import httpx
from tqdm.auto import tqdm

log = logging.getLogger(__name__)
__fname__ = pathlib.Path(__file__).stem
_max_download_tasks = int(os.getenv("MAX_DOWNLOAD_TASKS") or max(os.cpu_count(), 4))


def github_actor() -> str:
    res = os.getenv('GITHUB_ACTOR', None)
    if not res:
        res = getpass.getuser()
        assert res, "unable to get username"
        warnings.warn(f"No GITHUB_ACTOR env var provided => using \"{res}\"", UserWarning)
    return res


def hash_user_name(user: str) -> str:
    return hashlib.blake2b(user.encode('utf8'), digest_size=16, salt=b'maxisoft').hexdigest()

async def build_user_list(follower=True, following=True, *args, limit=1000):
    actor = github_actor()
    async with httpx.AsyncClient(headers={'Accept': 'application/vnd.github+json'},
                                 auth=(actor, os.getenv('GITHUB_TOKEN'))) as client:

        async def jget(url, **kwargs):
            r = await client.get(url, **kwargs)
            r.raise_for_status()
            return r.json()

        res = set()
        res.add(actor)
        if follower:
            res |= {d['login'] for d in await jget('https://api.github.com/user/followers?per_page=100')}
        if following:
            res |= {d['login'] for d in await jget('https://api.github.com/user/following?per_page=100')}

        res |= set(args)

        processed = set()
        processed.add(actor)
        tasks = []
        max_tasks = _max_download_tasks
        sem = asyncio.BoundedSemaphore(max_tasks)
        with tqdm(total=limit, unit='user') as pbar:
            pbar.update()

            async def get_user_follows(user: str):
                nonlocal res
                async with sem:
                    if follower:
                        res |= {d['login'] for d in
                                await jget(f'https://api.github.com/users/{user}/followers?per_page=100')}
                    if following:
                        res |= {d['login'] for d in
                                await jget(f'https://api.github.com/users/{user}/following?per_page=100')}
                return user

            while processed != res and len(processed) < limit:
                while len(tasks) >= max_tasks:
                    done, pending = await asyncio.wait(tasks, timeout=30, return_when=asyncio.FIRST_COMPLETED)
                    task: asyncio.Task

                    for task in done:
                        assert task.done()
                        user = task.result()
                        processed.add(user)
                        pbar.set_description_str(user)
                        pbar.update()
                        tasks.remove(task)

                    await asyncio.sleep(0)

                user = random.choice(list(res - processed))
                tasks.append(asyncio.ensure_future(get_user_follows(user)))

            await asyncio.gather(*tasks)

        res = sorted(res)
        res[res.index(actor)], res[0] = res[0], actor
        return res


_query_pattern = r'''
query _($user: String!){
  user(login: $user) {
    createdAt
    updatedAt
    followers {
      totalCount
    }
    following {
      totalCount
    }
    estimatedNextSponsorsPayoutInCents
    gistComments {
      totalCount
    }
    gists (privacy: PUBLIC){
      totalCount
    }
    issueComments {
      totalCount
    }
    issues {
      totalCount
    }
    organizations {
      totalCount
    }
    packages {
      totalCount
    }
    projects {
      totalCount
    }
    projectsV2 {
      totalCount
    }
    pullRequests {
      totalCount
    }
    repositories(privacy: PUBLIC) {
      totalCount
    }
    repositoriesContributedTo(privacy: PUBLIC) {
      totalCount
    }
    watching(privacy: PUBLIC) {
      totalCount
    }
    contributionsCollection {
      contributionYears
      pullRequestContributions {
        totalCount
      }
      issueContributions {
        totalCount
      }
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            contributionCount
            weekday
            date
          }
        }
      }
    }
  }
}
'''

graphql_api_rate = 5000  # assuming query weight is 1 (https://docs.github.com/en/graphql/overview/resource-limitations)


async def get_contributions(users: list[str]) -> dict[str, object]:
    actor = github_actor()
    sem = asyncio.BoundedSemaphore(_max_download_tasks)
    counter = 0
    async with httpx.AsyncClient(auth=(actor, os.getenv('GITHUB_TOKEN')), http2=True, timeout=30) as client:
        with tqdm(total=min(len(users), graphql_api_rate), unit='user contrib') as pbar:

            async def get_contributions(user: str):
                nonlocal counter
                query = _query_pattern
                async with sem:
                    start_time = time.monotonic()
                    counter += 1
                    if counter >= graphql_api_rate:
                        return None, None
                    try:
                        payload = {"query": query, "variables": {"user": user}}
                        r = await client.post('https://api.github.com/graphql', json=payload)
                        if r.status_code in (403, 502, 503):
                            log.warning("got temporary http error => retrying in 30s")
                            await asyncio.sleep(30)
                            r = await client.post('https://api.github.com/graphql', json=payload)
                        r.raise_for_status()
                    except httpx.HTTPError:
                        log.error("unable to complete http request for user %s", user, exc_info=True)
                        return None, None

                    while time.monotonic() - start_time < _max_download_tasks * 60 * 60 / graphql_api_rate:
                        log.debug("sleeping to avoid github rate limits")
                        await asyncio.sleep(.5)

                pbar.update()
                pbar.set_description_str(user)

                try:
                    return hash_user_name(user), r.json()
                except json.decoder.JSONDecodeError:
                    return None, None

            users = list(users)
            random.shuffle(users)
            tasks = [asyncio.ensure_future(get_contributions(user)) for user in users]

            res = dict(await asyncio.gather(*tasks))
            res.pop(None, None)
            return res


async def amain():
    log.info('creating user list')
    users = await build_user_list(limit=100)
    with open('users.json', 'w') as f:
        json.dump(users, f)

    log.info("gathering users' contributions")
    contributions = await get_contributions(users)

    contrib_path = pathlib.Path('contribs.json')
    if contrib_path.exists():
        log.info('loading previous contributions')
        prev_data = json.loads(contrib_path.read_bytes())
        contributions = prev_data | contributions

    with contrib_path.open('w') as f:
        json.dump(contributions, f, indent=2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(amain())
