from typing import Optional

import joblib
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.preprocessing import PowerTransformer, RobustScaler


def quantile_mask(data, lower=0.05, upper=0.95, **kwargs):
    mask = True
    if lower is not None and lower > 0:
        mask = data >= np.quantile(data, lower, interpolation='higher', **kwargs)
    if upper is not None and upper < 1:
        tmp = data <= np.quantile(data, upper, interpolation='lower', **kwargs)
        if mask is True:
            mask = tmp
        else:
            mask &= tmp
    return mask


def remove_outlier(df: pd.DataFrame):
    mask = np.ones(len(df), dtype=np.bool_)
    mask &= quantile_mask(df['age'].values, 0)
    mask &= df['age'].values > 0
    quantiles_columns = ['followers', 'following', 'gistComments', 'gists', 'issueComments', 'issues',
                         'organizations', 'pullRequests', 'repositories', 'repositoriesContributedTo', 'watching',
                         'pullRequestContributions', 'issueContributions', 'contrib_std']
    masks = quantile_mask(df[quantiles_columns].values, 0, axis=0, keepdims=True)
    mask &= np.all(masks, axis=1)
    mask &= quantile_mask(df['totalContributions'].values, 0.2, 0.8)
    mask &= df['totalContributions'].values > 0
    mask &= quantile_mask(df['contrib_skew'].values)
    mask &= quantile_mask(df['repositoriesContributedTo'].values, 0.2)
    mask &= df['repositoriesContributedTo'].values > 1

    return df.loc[mask], mask


def normalize_users(df: pd.DataFrame, transformer: Optional[PowerTransformer] = None, max_value=3.3):
    scale_with_age_columns = ['followers', 'following', 'gistComments', 'gists', 'issueComments', 'issues',
                              'pullRequests', 'repositories',
                              'repositoriesContributedTo', 'watching', 'pullRequestContributions', 'issueContributions']
    df[scale_with_age_columns] /= np.log1p(df['age'].values[:, None])
    if transformer is None:
        transformer = PowerTransformer(copy=False)
        transformer.fit_transform(df.values)
    else:
        transformer.copy = False
        transformer.transform(df.values)
    np.nan_to_num(df.values, copy=False, posinf=max_value, neginf=-max_value)
    df /= max_value
    np.tanh(df.values, out=df.values)
    df *= max_value
    return df, transformer


def normalize_weeks(weeks: np.ndarray, transformer: Optional[RobustScaler] = None, max_value=5.0):
    shape = tuple(weeks.shape)
    if len(weeks.shape) != 1:
        weeks = weeks.reshape(-1, 1)
    weeks = np.log1p(weeks, out=weeks)
    if transformer is None:
        transformer = RobustScaler(copy=False, with_centering=False, unit_variance=True)
        transformer.fit_transform(weeks)
    else:
        transformer.copy = False
        transformer.transform(weeks)

    weeks /= max_value
    np.tanh(weeks, out=weeks)
    weeks *= max_value
    weeks = weeks.reshape(shape)
    return weeks, transformer


def augment(df: pd.DataFrame, weeks: np.ndarray):
    contrib_per_day = weeks.reshape((weeks.shape[0], -1, 7))
    contrib_per_week = contrib_per_day.sum(axis=-1)

    # df['contrib_mean'] = np.mean(contrib_per_week, axis=-1)
    df['contrib_std'] = np.std(contrib_per_week, axis=-1)
    df['contrib_skew'] = scipy.stats.skew(contrib_per_week, axis=-1)

    contrib_per_day_mean = np.mean(contrib_per_day, axis=1)
    contrib_per_day_std = np.std(contrib_per_day, axis=1)
    contrib_per_day_skew = scipy.stats.skew(contrib_per_day, axis=1)

    for i in range(contrib_per_day.shape[-1]):
        df[f'contrib_d{i}_mean'] = contrib_per_day_mean[:, i]
        df[f'contrib_d{i}_std'] = contrib_per_day_std[:, i]
        df[f'contrib_d{i}_skew'] = contrib_per_day_skew[:, i]

    contrib_fft = np.fft.rfft(weeks)

    top_args = np.argpartition(np.absolute(contrib_fft), -7, axis=-1)[:, -7:]
    top_values = np.take_along_axis(contrib_fft, top_args, axis=-1)
    for i in range(top_args.shape[-1]):
        df[f'contrib_fft{i}_real'] = top_values.real[:, i]
        df[f'contrib_fft{i}_imag'] = top_values.imag[:, i]
        df[f'contrib_fft{i}_argmax'] = top_args[:, i]

    return df.astype('float32', copy=True)


def main(report=True, inference=False, user_transformer=None, week_transformer=None):
    userdata = np.load('userdata.npz')
    users = userdata['users']
    user_columns = userdata['user_columns']
    weeks = userdata['weeks']

    user_df = pd.DataFrame(users, userdata['indices'], columns=user_columns)

    user_df = augment(user_df, weeks)
    if not inference:
        user_df, mask = remove_outlier(user_df)
    else:
        mask = np.ones(len(user_df), dtype=np.bool_)
    user_df, transformer = normalize_users(user_df, user_transformer)
    zero_df = user_df.iloc[[0], :] * 0
    zero_df, _ = normalize_users(zero_df, transformer)
    zero_df.values[:] = np.where(zero_df == 0, np.min(user_df.values, axis=0, keepdims=True), zero_df.values)

    from sklearn.cluster import KMeans

    km = KMeans(6)
    km.cluster_centers_ = np.asfarray([0, 1, 2, 4, 6, 12])[:, None]
    km._n_threads = getattr(km, '_n_threads', None)
    weeks_cluster = km.predict(np.asfarray(weeks[mask, :]).reshape(-1, 1))
    print(pd.Series(weeks_cluster).value_counts())
    scaled_weeks, w_transformer = normalize_weeks(np.asfarray(np.copy(weeks)), week_transformer)

    np.savez_compressed("ml",
                        weeks=weeks[mask, :].astype('float32'),
                        scaled_weeks=scaled_weeks[mask, :].astype('float32'),
                        users=user_df.to_numpy('float32'),
                        columns=np.asarray(user_df.columns),
                        zeros=zero_df.to_numpy('float32'),
                        weeks_cluster=weeks_cluster.reshape(-1, weeks.shape[-1]),
                        value_counts=pd.Series(weeks_cluster).value_counts().to_numpy('int32'),
                        cluster_centers=km.cluster_centers_
                        )

    joblib.dump({'user_scaler': transformer, 'weeks_scaler': w_transformer}, 'scalers.pkl.z')
    if report:
        from pandas_profiling import ProfileReport
        profile = ProfileReport(user_df, interactions={'targets': [], 'continuous': False},
                            plot={'histogram': {'bins': 16}})
        profile.to_file("userdata_report.html")


if __name__ == '__main__':
    main()
