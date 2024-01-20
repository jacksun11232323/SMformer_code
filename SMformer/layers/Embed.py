import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from models import sim
from torch import (randn)
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

def FFT_for_Period(x, k=2):
    # [B, T, C]
   # x = x.permute(0,2,1)
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    top_list = torch.argsort(frequency_list, descending=True)[:k]
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class PatchEmbedding(nn.Module):
    def __init__(self,configs, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        #自己加的
        self.predict_linear = nn.Linear(
            configs.seq_len, configs.pred_len + configs.seq_len)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.back_linear = nn.Linear(configs.d_model, 7)
        self.lahui = nn.Linear(96,32)
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(32, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        #遮蔽率获得
        self.mask_ratio = configs.mask_ratio
        self.mask_token = nn.Parameter(randn(configs.d_model))
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        #自己加的
        self.sep_len = configs.seq_len
    def forward(self, x, x_mark):
        # do patching
        #n_vars = x.shape[1]
        # 修改从这里开始 x(32,7,96)
        period_list, _ = FFT_for_Period(x,k=5)
        #求解应该在尾部填补多少步0
        period = period_list[0]
        remainder = x.shape[-1] % period
        if(remainder == 0):
            res_padnum = 0
        else:
            res_padnum = period - remainder
        x = F.pad(x, (0, res_padnum), value=0)
        #相当于在最后一个维度，以patch_len为窗口大小，stride为步长，这样去滑动进行分隔
        x = x.unfold(dimension=-1, size=period, step=period)#()
        # x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # if(x.shape[-1] <= 16):
        #     x = F.pad(x,(0,16-x.shape[-1]),value=0)
        # Input encoding
        #x = self.enc_embedding(x,x_mark)
        #由于为了方便后期的处理，我们在这个地方对维度进行一统一化处理,我们块内采用最后的值填充
        if(x.size(-1) <= 32 ):
            in_padnum = 32 - x.size(-1)
            #(32,64c,patch_num,32)
            x = F.pad(x,(0,in_padnum))
        else:
            in_padnum = 96 - x.size(-1)
            #(32,64,patch_num,96)
            x = F.pad(x, (0, in_padnum))
        x = x.permute(0,3,2,1)#(32，32，48，64) (B,patch_len,num_patch,c)
        # x = self.back_linear(x)#(32,32,48,7)
        x = x.permute(0,3,2,1)#(b,c,patch_num,patch_len)
        if(x.size(-1) == 96):
            x = self.lahui(x)
        #x = x.permute(0,2,1,3)#(bs,patch_num,c,patch_len)
        x= torch.reshape(x,(x.shape[0] * x.shape[1] , x.shape[2],x.shape[3]))
        #对patch内进行embeding以及位置编码
        x = self.value_embedding(x) + self.position_embedding(x)#(224,48,64)
        x = self.dropout(x)
        y = x
        #在这进行遮蔽
        n_masked_tokens = int(self.mask_ratio * x.shape[1])
        mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1], 1)
        bitmask = sim.get_bitmask(
            batch_size=x.shape[0],
            n_tokens=x.shape[1],
            n_masked_tokens=n_masked_tokens,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        x = sim.do_mask_tokens(
            tokens=x,
            mask_tokens=mask_tokens,
            bitmask=bitmask,
        )
        bitmask = bitmask.cpu()
        bitmask = np.repeat(bitmask, period, axis=1)
        bitmask = bitmask.bool()
        if (bitmask.size(-1) > self.sep_len):
            bitmask = bitmask[:, :self.sep_len]
        return x , y , bitmask,period

