import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from layers.Embed import DataEmbedding
import torch.nn.functional as F
import numpy as np

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):


    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.patch_embedding = PatchEmbedding(
            configs,configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder_masken = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder_shuffle = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        #shuflle率
        self.shuffle_ratio = configs.shuffle_ratio
        self.total_num = int(configs.shuffle_ratio * configs.seq_len)
        #自己添加的网络
        self.backlinear = nn.Linear(96,64)
        self.shuffle_linear = nn.Linear(configs.enc_in,configs.seq_len)
        # Prediction Head
        self.head_nf = 64 * configs.d_model
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head1 = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
            self.head2 = FlattenHead(configs.enc_in, self.head_nf, int(configs.pred_len * configs.shuffle_ratio),
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        n_vars = x_enc.shape[-1]
        # do patching and embedding
        # x_enc = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out , y , bit_mask, period = self.patch_embedding(x_enc ,x_mark_enc)
        # 进行shuflle操作
        dim_size = y.size(1)
        # 选取遮蔽率的元素数量
        num_elements = int(dim_size * self.shuffle_ratio)
        if num_elements == 1:
            num_elements = 2
        # 对应于96步进行shuffle的位置
        shuffle_count = num_elements * period
        # 创建一个布尔张量，表示要进行洗牌的元素
        shuffle_mask = torch.zeros(dim_size).bool()
        indices = torch.randperm(dim_size - 1)[:num_elements]
        shuffle_mask[indices] = True
        # 获取需要洗牌的元素
        shuffled_elements = y[:, shuffle_mask]

        # 获取不需要洗牌的元素
        non_shuffle_elements = y[:, ~shuffle_mask]
        # 对洗牌的元素进行随机重排
        perm = torch.arange(num_elements)
        shuffled_indices = torch.randperm(num_elements)
        if num_elements != 0 and num_elements != 1:
            while torch.all(torch.eq(perm, shuffled_indices)):
                shuffled_indices = torch.randperm(num_elements)
        shuffled_elements = shuffled_elements[:, shuffled_indices]
        # 将洗牌后的元素放回原张量中
        y[:, shuffle_mask] = shuffled_elements
        shuffle_mask = shuffle_mask.cpu()
        shuffle_mask = np.repeat(shuffle_mask, period)
        shuffle_mask = shuffle_mask.bool()
        if (shuffle_mask.size(0) > self.seq_len):
            shuffle_mask = shuffle_mask[:self.seq_len]
        # 对齐维度
        if shuffle_count > self.total_num:
            num = int(shuffle_count - self.total_num)
            last_true_indices = (shuffle_mask == True).nonzero()[-num:].squeeze()
            shuffle_mask[last_true_indices] = False
        if shuffle_count < self.total_num:
            num = int(self.total_num - shuffle_count)
            last_true_indices = (shuffle_mask == False).nonzero()[-num:].squeeze()
            shuffle_mask[last_true_indices] = True
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out_mask, attns = self.encoder_masken(enc_out)
        enc_out_shuffle, attns = self.encoder_shuffle(y)
        # z: [bs x nvars x patch_num x d_model]
        enc_out_mask = torch.reshape(
            enc_out_mask, (-1, n_vars, enc_out_mask.shape[-2], enc_out_mask.shape[-1]))
        enc_out_shuffle = torch.reshape(
            enc_out_shuffle, (-1, n_vars, enc_out_shuffle.shape[-2], enc_out_shuffle.shape[-1]))
        # z: [bs x nvars x d_model x patch_num](32,7,64,48)
        enc_out_mask = enc_out_mask.permute(0, 1, 3, 2)
        enc_out_shuffle = enc_out_shuffle.permute(0, 1, 3, 2)
        #把维度对齐，由于按照周期打，所以patch的个数不同，所以进行维度对其
        pad_num = 96 - enc_out_mask.size(-1)
        #(32,7,64,96)
        enc_out_mask = F.pad(enc_out_mask, (0, pad_num),value=0)
        enc_out_shuffle = F.pad(enc_out_shuffle, (0, pad_num), value=0)
        enc_out_mask = self.backlinear(enc_out_mask)
        enc_out_shuffle = self.backlinear(enc_out_shuffle)

        # Decoder
        dec_out_mask = self.head1(enc_out_mask)  # z: [bs x nvars x target_window]
        dec_out_shuffle = self.head2(enc_out_shuffle)
        dec_out_mask = dec_out_mask.permute(0, 2, 1)
        dec_out_shuffle = dec_out_shuffle.permute(0, 2, 1)
        dec_out_shuffle = self.shuffle_linear(dec_out_shuffle)
        dec_out_shuffle = dec_out_shuffle.permute(0,2,1)

        # De-Normalization from Non-stationary Transformer
        dec_out_mask = dec_out_mask * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out_shuffle = dec_out_shuffle * \
        #                (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out_mask = dec_out_mask + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out_shuffle = dec_out_shuffle + \
        #                (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out_mask, dec_out_shuffle,bit_mask,shuffle_mask




    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out_mask ,dec_out_shuffle , bit_mask,shuffle_mask = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out_mask[:, -self.pred_len:, :] ,dec_out_shuffle, bit_mask,shuffle_mask# [B, L, D]
        return None
