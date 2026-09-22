import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import torch.nn.functional as F
from layers.Embed import PositionalEmbedding , DataEmbedding_inverted

def find_period(x, k=1):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0:2] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    periods = x.shape[1] // top_list
    return periods

def find_harmonic_periods(x, top_k, cycle_len):
    max_peaks = x.shape[1] // 2 - 1
    # Get enough dominant periods from the FFT
    periods = find_period(x, k=max_peaks)

    # Keep only harmonic periods
    harmonic_periods = [
        int(p) for p in periods
        if p > 1 and cycle_len % int(p) == 0 and int(p) < cycle_len
    ]

    # Return the top_k harmonic periods
    return harmonic_periods[:top_k]

class PeriodMix(nn.Module):
    def __init__(self, seq_len, patch_len):
        super(PeriodMix, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.mixer = nn.Linear(seq_len+seq_len, seq_len)
        
    def forward(self, cycle, periods):
        res = cycle
        i = len(periods)-1
        while i >= 0:
            cycle = torch.cat([cycle, periods[i]], dim=-1)
            cycle = self.mixer(cycle)
            i-=1
        ex_cycle = res + cycle
        en_cycle = ex_cycle.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        en_cycle = en_cycle.reshape(ex_cycle.shape[0]*ex_cycle.shape[1], en_cycle.shape[2], -1)
        return en_cycle, ex_cycle
        
    
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
    

def conv3x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=3, stride=stride, padding=1, bias=False)

class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Sequential(
            conv3x1(in_channels=1, out_channels=d_model // 8, stride= 2),
            nn.GELU(),
            conv3x1(in_channels=d_model // 8, out_channels=d_model // 4, stride=2),
            nn.GELU(),
            conv3x1(in_channels=d_model // 4, out_channels=d_model // 2, stride=2),
            nn.GELU(),
            conv3x1(in_channels=d_model // 2, out_channels=d_model, stride=2)
        )
        
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        batch,n_vars,temp = x.shape
        x = x.reshape(batch*n_vars, 1, temp)
        # Input encoding
        x = self.value_embedding(x)
        x = x.transpose(1, 2)        # [B*C, N, d_model]
        x = x + self.position_embedding(x)
        return self.dropout(x),batch,n_vars,x.shape[2]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_core, channels, seq_len, patch_len, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.channels = channels
        self.d_core = d_core
        self.proj1 = nn.Linear(patch_len, d_model)
        self.proj2 = nn.Linear(seq_len, d_model)
        self.gen1 = nn.Linear(d_model, d_model)
        self.gen2 = nn.Linear(d_model, d_core)
        self.gen3 = nn.Linear(d_model + d_core, d_model)
        self.gen4 = nn.Linear(d_model, d_core)
        self.gen5 = nn.Linear(d_model + d_model + d_core, d_model)
        self.gen6 = nn.Linear(d_model, d_model)
        self.gen7 = nn.Linear(d_model + d_model + d_model, d_model)
        self.gen8 = nn.Linear(d_model, d_model)
        self.en_mix = nn.Linear(d_model + d_model, d_model)
        self.ex_mix = nn.Linear(d_model + d_model, d_model)
        
        
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, ex, en_cycle, ex_cycle, attn_mask=None, tau=None, delta=None):
        
        bc, p, d_model = x.shape
        
        en_cycle = F.gelu(self.proj1(en_cycle))
        ex_cycle = F.gelu(self.proj2(ex_cycle))
        initial = F.gelu(self.gen1(x))
        en_cycle_mix = torch.cat([initial, en_cycle], dim=-1)
        core = F.gelu(self.en_mix(en_cycle_mix))
        core = self.gen2(core)
        

        # stochastic pooling
        if self.training:
            ratio = F.softmax(core, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, p)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(bc, -1, 1).permute(0, 2, 1)
            core = torch.gather(core, 1, indices)
        else:
            weight = F.softmax(core, dim=1)
            core = torch.sum(core * weight, dim=1, keepdim=True)
        
        # scores = self.score(core)                       # bc × p × 1
        # weights = F.softmax(scores, dim=1)              # bc × p × 1
        # core = (core * weights).sum(dim=1, keepdim=True)  # bc × 1 × d_core
        
        core = core.reshape(-1,self.channels,1,self.d_core).squeeze(dim=2)
        b, c, _ = core.shape
        core = torch.cat([ex, core], dim=2)
        core = F.gelu(self.gen3(core))
        ex_cycle_mix = torch.cat([ex_cycle, core], dim=-1)
        ex_core = F.gelu(self.ex_mix(ex_cycle_mix))
        
        ex_core = self.gen4(ex_core)
        
        # stochastic pooling
        if self.training:
            ratio = F.softmax(ex_core, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, self.channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(b, -1, 1).permute(0, 2, 1)
            ex_core = torch.gather(ex_core, 1, indices)
            ex_core = ex_core.repeat(1, self.channels, 1)
        else:
            weight = F.softmax(ex_core, dim=1)
            ex_core = torch.sum(ex_core * weight, dim=1, keepdim=True).repeat(1, self.channels, 1)
        
        
        ex_combined = torch.cat([ex_cycle_mix, ex_core], dim=2)
        ex_combined = F.gelu(self.gen5(ex_combined))
        ex_combined = self.gen6(ex_combined)
        
        ex_combined = ex_combined.unsqueeze(dim=2)
        ex_combined = ex_combined.reshape(b*c, 1, -1).repeat(1, p, 1)
        combined_cat = torch.cat([en_cycle_mix, ex_combined], dim=2)
        combined_cat = F.gelu(self.gen7(combined_cat))
        combined_cat = self.gen8(combined_cat)

        x = x + self.dropout(combined_cat)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, ex, en_cycle, ex_cycle, x_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, ex, en_cycle, ex_cycle, attn_mask=x_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
    

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.cycle_len = configs.cycle
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.k = configs.topK
        
        self.data = torch.nn.Parameter(torch.zeros(self.cycle_len, configs.n_vars), requires_grad=True)
        
        self.harmonic_data = nn.ParameterList([
            nn.Parameter(torch.zeros(min(self.cycle_len, self.seq_len) // 2, configs.n_vars))
            for _ in range(self.k)
        ])
        
        self.period_mix = PeriodMix(configs.seq_len, configs.patch_len)
        self.en_Embedding = EnEmbedding(configs.n_vars, configs.d_model, configs.patch_len, configs.dropout)
        #self.data = torch.nn.Parameter(torch.zeros(self.cycle_len, configs.n_vars), requires_grad=True)
        # Embedding
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model,
                    configs.d_core,
                    configs.n_vars,
                    configs.seq_len,
                    configs.patch_len,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
        )

        self.head_nf = configs.d_model * (self.patch_num)
        self.head = FlattenHead(configs.n_vars, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)
        # Decoder
        #self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, cycle_index, x_mark_enc, x_dec, x_mark_dec):
        
        
        # dom = find_period(x_enc, k=5)
        periods = find_harmonic_periods(x_enc, self.k, self.cycle_len)
        # print(periods)
        
        
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, seq_len, N = x_enc.shape
        
        
        
        gather_index = (cycle_index.view(-1, 1) + torch.arange(seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        
        cycle_indices = [
            cycle_index % p
            for p in periods
        ]
        gather_indices = [
            (cycle_indices[i].view(-1, 1) + torch.arange(seq_len, device=cycle_index.device).view(1, -1)) % periods[i]
            for i in range(len(periods))
        ]  
        harmonic_cycles = [
            self.harmonic_data[i][gather_indices[i]].permute(0,2,1)  
            for i in range(len(periods))
        ]
        
        cycle = self.data[gather_index].permute(0,2,1)
        if self.k > 0 :
            en_cycle, ex_cycle = self.period_mix(cycle, harmonic_cycles)
        else :
            ex_cycle = cycle
            en_cycle = ex_cycle.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
            en_cycle = en_cycle.reshape(ex_cycle.shape[0]*ex_cycle.shape[1], en_cycle.shape[2], -1)
            
        enc_out, batch, channel, _ = self.en_Embedding(x_enc.permute(0,2,1))
        ex_out = self.ex_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, ex_out, en_cycle, ex_cycle)
        dec_out = self.head(enc_out).reshape(batch, channel, -1)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, cycle_index, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, cycle_index, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]