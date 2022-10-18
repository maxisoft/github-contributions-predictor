import asyncio
import dataclasses
import datetime
import importlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch import nn

logger = logging.getLogger(__name__)

torch.set_num_threads(os.cpu_count())

gather_data = importlib.import_module('0-gather_data')
pack_data = importlib.import_module('1-pack-data')
preprocess = importlib.import_module('2-preprocess')


class SwitchableNorm1d(nn.Module):
    """Simplification over https://arxiv.org/abs/1907.10473
    by using only batchnorm and layernorm.
    Allowing the layer to work with both conv data shape (N, C, H) and linear shape (N, *, F)
    """

    def __init__(self, num_features, normalized_shape=None, weight_shape=None, momentum=0.1, bias=True):
        super().__init__()
        if normalized_shape is None:
            normalized_shape = num_features
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)
        self.ln = nn.LayerNorm(normalized_shape)
        if weight_shape is None:
            weight_shape = normalized_shape
            if isinstance(normalized_shape, int) or len(normalized_shape) == 1:
                weight_shape = (num_features,)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        torch.nn.init.normal_(self.weight, 0, 0.7)
        if bias:
            self.bias = nn.Parameter(torch.zeros(weight_shape))
        else:
            self.bias = None

    def forward(self, input: Tensor):
        bn = self.bn(input)
        ln = self.ln(input)
        weight = self.weight.sigmoid()
        if weight.dim() < input.dim() and weight.size(-1) != input.size(-1):
            # assuming to work with convolution shape (N, C, H, *)
            pad = tuple(1 for _ in range(input.dim() - 1 - weight.dim()))
            weight = weight.view(1, *weight.shape, *pad)
            assert weight.dim() == input.dim()
        res = weight * bn + (1 - weight) * ln
        if self.bias is not None:
            res = res + self.bias
        return res


class GCModel(nn.Module):
    def __init__(self, seq_size: int, user_features: int):
        super().__init__()
        hidden_size = 256

        self.contrib_w = nn.Parameter(torch.empty((seq_size, user_features, 8), dtype=torch.cfloat))
        torch.nn.init.kaiming_normal_(self.contrib_w)

        self.contrib_b = nn.Parameter(torch.empty((7, seq_size, user_features, 8), dtype=torch.cfloat))
        torch.nn.init.xavier_uniform_(self.contrib_b, 0.5)

        self.contrib_alpha = nn.Parameter(torch.empty((1, 1, user_features), dtype=torch.cfloat))
        torch.nn.init.kaiming_normal_(self.contrib_alpha)

        self.hidden_alpha = nn.Parameter(torch.empty((7, user_features), dtype=torch.cfloat))
        torch.nn.init.kaiming_normal_(self.hidden_alpha)

        self.user_bn = SwitchableNorm1d(user_features, momentum=0.01)

        self.hidden = nn.Sequential(
            nn.Conv1d(user_features, hidden_size, 1),
            nn.Mish(inplace=True),
            nn.Conv1d(hidden_size, hidden_size, 7, 7),
            SwitchableNorm1d(hidden_size, seq_size // 7, (hidden_size, 1)),
            nn.Mish(inplace=True),
            nn.Conv1d(hidden_size, user_features, 3)
        )

        self.final = nn.Sequential(
            SwitchableNorm1d(7 * user_features, momentum=0.01),
            nn.Linear(7 * user_features, hidden_size),
            nn.Mish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 7 * 6 * 2, bias=True)
        )

    def forward(self, day: Tensor, users: Tensor, contribs: Tensor):
        contrib_b = self.contrib_b[None, :, :, :, :].expand(day.size(0), -1, -1, -1, -1)
        contrib_b = torch.gather(contrib_b, 1, day[:, :, None, None, None].expand(-1, 1, *contrib_b.size()[2:]))

        if self.training:
            contrib_mask = torch.rand_like(contribs) > torch.randint(0, 5, (contribs.size(0), 1),
                                                                     device=contribs.device, dtype=contribs.dtype) / 100
            contribs_std, contribs_mean = torch.std_mean(contribs)
            contribs_std2, contribs_mean2 = torch.std_mean(contribs, dim=1, keepdim=True)

            contribs_std = torch.where(
                torch.randint_like(contribs, 0, 3) == 0,
                contribs_std,
                contribs_std2
            )
            contribs_mean = torch.where(
                torch.randint_like(contribs, 0, 3) == 0,
                contribs_mean,
                contribs_mean2
            )

            noise = torch.normal(contribs_mean, contribs_std).to(device=contribs.device)
            mult = torch.randint_like(noise, 0, 101).float() / 100
            contribs = torch.where(
                torch.randint_like(contribs, 0, 11) > 9,
                (1 - mult) * contribs + noise * mult,
                contribs + (noise - contribs_mean) * torch.randint_like(noise, 0, 5).float() / 100
            )
            contribs = contribs * contrib_mask
            contribs = contribs.relu()

        contribs_x = torch.fft.fft(contribs[:, :, None, None], dim=1,
                                   norm='ortho') * self.contrib_w + contrib_b.squeeze_(1)

        users = self.user_bn(users)
        if self.training:
            users_std = torch.std(users, dim=0, keepdim=True).expand_as(users)
            noise = torch.normal(0, users_std).to(device=users.device)
            users = users + noise.detach() * torch.randint_like(noise, 0, 5) / 100
        contribs_x = torch.fft.fft(users, dim=1, norm='ortho')[:, None, :, None].expand_as(contribs_x) * torch.fft.fft(
            contribs_x, dim=3)
        contribs_x = torch.fft.ifft(contribs_x, dim=3)
        contrib_alpha = torch.sigmoid(self.contrib_alpha)
        contribs_x = contrib_alpha * torch.amax(torch.abs(contribs_x), dim=3) + (1 - contrib_alpha) * torch.mean(
            contribs_x, dim=3)
        acontribs_x = torch.abs(contribs_x)
        contribs_x = contribs_x * (
                    acontribs_x > torch.quantile(acontribs_x, 0.1, dim=1, keepdim=True))  # mimic fft filters
        contribs_x = contribs_x * (acontribs_x > torch.quantile(acontribs_x, 0.1, dim=2, keepdim=True))  # on both axis
        contribs_x = torch.fft.ifft2(contribs_x, dim=(1, 2), norm='ortho')
        hidden = self.hidden(contribs_x.real.permute(0, 2, 1)).permute(0, 2, 1)
        hidden_alpha = self.hidden_alpha.tanh()
        hidden = (torch.fft.fft2(hidden, dim=(2, 1)) * hidden_alpha +
                  torch.fft.fft(contribs_x[:, -hidden.size(1):, :], dim=1) * (1 - hidden_alpha))
        ahidden = torch.abs(hidden)
        hidden = hidden * (ahidden > torch.quantile(ahidden, 0.3, dim=2, keepdim=True))
        hidden = torch.fft.ifft(hidden, dim=1)
        res = self.final(hidden.real.reshape(hidden.size(0), -1)).view(hidden.size(0), 6, 7, 2)
        return res


_default_users = (gather_data.github_actor(),)


@dataclasses.dataclass()
class UserDataValue:
    index: int
    user: str


def sanitize_path(path: str) -> str:
    stable = False
    res = path
    while not stable:
        tmp = res.replace('..', '_').replace('/', '_')
        stable = tmp == res
        res = tmp
    return res

async def amain(users=_default_users):
    original_path = Path().resolve()
    with tempfile.TemporaryDirectory('gai') as tmp:
        os.chdir(tmp)
        contributions: dict[str, object] = await gather_data.get_contributions(users)
        with Path('contribs.json').open('w') as f:
            json.dump(contributions, f)

        reorder_indices = np.arange(len(contributions), dtype=np.int64)
        user_data: Dict[str, UserDataValue] = {gather_data.hash_user_name(user): UserDataValue(i, user) for
                                               i, user in enumerate(users)}
        for i, user_hash in enumerate(contributions.keys()):
            reorder_indices[i] = user_data[user_hash].index

        try:
            pack_data.pack_data(align_weekday=False)
        except Exception as e:
            logger.error("", exc_info=True)
            raise
        scalers = joblib.load(original_path / 'scalers.pkl.z')
        preprocess.main(report=False, inference=True, user_transformer=scalers['user_scaler'],
                        week_transformer=scalers['weeks_scaler'])

        with np.load('ml.npz') as data:
            users = data['users']
            weeks = data['scaled_weeks']
        now = datetime.datetime.utcnow()
        weekday = now.weekday()
        weeks = weeks[:, :-1]

        day = abs((weekday + 1) % 7)

        model = GCModel(64, users.shape[-1])
        model.load_state_dict(torch.load(original_path / 'model.pth', 'cpu'))
        model.eval()

        with torch.no_grad():
            result = model(torch.tensor([day])[None, :].expand(len(users), -1), torch.from_numpy(users),
                           torch.from_numpy(weeks[:, -64:]))
            am = result[:, :, :, 1].argmax(1, keepdims=True)
            contribs = torch.gather(result[:, :, :, 0], 1, am)
            print(contribs.numpy())

        os.chdir(original_path)
        out_path = (original_path / "out")
        out_path.mkdir(exist_ok=True)
        date_array = pd.to_datetime([now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=i) for i in range(contribs.size(-1))])
        for i, user_hash in enumerate(contributions.keys()):
            csv_path = out_path / (sanitize_path(user_data[user_hash].user) + '.csv')
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=['date'], index_col=['date'])
            else:
                df = pd.DataFrame(columns=['date', 'contrib_count'])
                df.set_index('date', inplace=True)

            new_df = pd.DataFrame(np.round(contribs[i, :, :].numpy().transpose()), index=date_array, columns=['contrib_count'])
            df = df.append(new_df)
            df = df[~df.index.duplicated(keep='last')] # remove duplicate if any
            df.sort_index(inplace=True)

            df.to_csv(csv_path, index_label='date')
        await asyncio.sleep(1)


if __name__ == '__main__':
    try:
        asyncio.run(amain())
    except:
        logger.error("", exc_info=True)
        raise
