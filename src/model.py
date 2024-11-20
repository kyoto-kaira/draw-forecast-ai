import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MODEL_CONFIG = {
    "input_dim": 5,
    "emb_dim": None,
    "enc_hid_dim": 512,
    "enc_num_layers": 1,
    "latent_dim": 128,
    "dec_hid_dim": 2048,
    "dec_num_layers": 1,
    "drop_prob": 0.1,
    "num_mixture": 20,
    "KL_min": 0.2,
    "eta_start": 0.01,
}

class LayerNormLSTMCell(nn.Module):
    def __init__(self, num_inputs, num_hidden, drop_prob=0.1, forget_gate_bias=-1):
        super(LayerNormLSTMCell, self).__init__()

        self.forget_gate_bias = forget_gate_bias
        self.num_hidden = num_hidden
        self.fc_i2h = nn.Linear(num_inputs, 4 * num_hidden)
        self.fc_h2h = nn.Linear(num_hidden, 4 * num_hidden)

        self.ln_i2h = nn.LayerNorm(4 * num_hidden)
        self.ln_h2h = nn.LayerNorm(4 * num_hidden)
        self.drop = nn.Dropout(p=drop_prob)

        self.ln_h2o = nn.LayerNorm(num_hidden)

    def forward(self, inputs, state=None):
        if state:
            hx, cx = state
        else:
            zeros = torch.zeros(
                inputs.size(0),
                self.num_hidden,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            hx, cx = zeros, zeros
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)
        x = self.ln_i2h(i2h) + self.ln_h2h(h2h)  # layernorm
        gates = x.split(self.num_hidden, 1)

        in_gate = F.sigmoid(gates[0])
        forget_gate = F.sigmoid(gates[1] + self.forget_gate_bias)
        out_gate = F.sigmoid(gates[2])
        in_transform = F.tanh(gates[3])
        in_transform = self.drop(in_transform)  # dropout

        cx = forget_gate * cx + in_gate * in_transform
        hx = out_gate * F.tanh(self.ln_h2o(cx))  # layernorm
        return (hx, cx)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=5,
        emb_dim=None,
        hid_dim=256,
        num_layers=1,
        drop_prob=0.1,
        latent_dim=128,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hid_dim
        self.emb = nn.Linear(input_dim, emb_dim) if emb_dim else nn.Identity()
        if not emb_dim:
            emb_dim = input_dim
        self.lstm_for = LayerNormLSTMCell(emb_dim, hid_dim, drop_prob)
        self.lstm_back = LayerNormLSTMCell(emb_dim, hid_dim, drop_prob)
        self.linear1 = nn.Linear(hid_dim * 2, latent_dim)
        self.linear2 = nn.Linear(hid_dim * 2, latent_dim)

    def forward(self, x, seq_len=None, is_eval=False):
        state1, state2 = None, None
        x = self.emb(x)
        if is_eval:
            for i in range(x.shape[1]):
                state1 = self.lstm_for(x[:, i, :], state1)
                state2 = self.lstm_back(x[:, -i - 1, :], state2)
            h1, _ = state1
        else:
            h1 = torch.zeros(x.shape[0], self.hidden_size).to(x.device)
            seq_back_len = x.shape[1] - 1 - seq_len
            for i in range(x.shape[1]):
                state1 = self.lstm_for(x[:, i, :], state1)
                state2 = self.lstm_back(x[:, -i - 1, :], state2)
                if i in seq_len:
                    tmp, _ = state1
                    h1[torch.where(seq_len == i)[0], :] = tmp[
                        torch.where(seq_len == i)[0], :
                    ].squeeze(0)
                if i in seq_back_len:
                    hx, cx = state2
                    hx, cx = hx.clone(), cx.clone()
                    hx[torch.where(seq_back_len == i)[0], :] = torch.zeros(
                        self.hidden_size, device=x.device
                    )
                    cx[torch.where(seq_back_len == i)[0], :] = torch.zeros(
                        self.hidden_size, device=x.device
                    )
                    state2 = (hx, cx)
        h2, _ = state2
        x = torch.cat([h1, h2], dim=-1)
        x = x.reshape(-1, self.hidden_size * 2)
        return self.linear1(x), self.linear2(x).mul(0.5).exp_()


class Decoder(nn.Module):
    def __init__(
        self, latent_dim=128, hid_dim=256, num_layers=1, drop_prob=0.1, num_mixture=20
    ):
        super(Decoder, self).__init__()
        self.lstm = LayerNormLSTMCell(latent_dim + 5, hid_dim, drop_prob)
        self.output_layer = nn.Linear(hid_dim, 3 + num_mixture * 6)
        self.hid_layer = nn.Linear(latent_dim, hid_dim)
        self.memory_layer = nn.Linear(latent_dim, hid_dim)

    def forward(self, z, inputs, state=None, is_eval=False):
        if is_eval:
            decoder_input = torch.cat([inputs, z], dim=-1)
            state = self.lstm(decoder_input, state)
            output = self.output_layer(state[0])
            return output, state
        else:
            sequence_len = inputs.size(-2)
            state = (F.tanh(self.hid_layer(z)), F.tanh(self.memory_layer(z)))
            z = z.unsqueeze(1).repeat(1, sequence_len, 1)
            decoder_inputs = torch.cat([inputs, z], dim=-1)
            output = []
            for i in range(sequence_len):
                decoder_input = decoder_inputs[:, i, :]
                state = self.lstm(decoder_input, state)
                h, _ = state
                output.append(h)
            output = torch.stack(output, dim=1)
            output = self.output_layer(output)
            return output, state


class SketchRNN(nn.Module):
    def __init__(
        self,
        device,
        input_dim=5,
        emb_dim=None,
        enc_hid_dim=256,
        enc_num_layers=1,
        latent_dim=128,
        dec_hid_dim=256,
        dec_num_layers=1,
        drop_prob=0.1,
        num_mixture=20,
        KL_min=0.2,
        eta_start=0.01,
    ):
        super(SketchRNN, self).__init__()
        self.device = device
        self.KL_min = torch.tensor(KL_min)
        self.eta = eta_start
        self.encoder = Encoder(
            input_dim, emb_dim, enc_hid_dim, enc_num_layers, drop_prob, latent_dim
        )
        self.decoder = Decoder(
            latent_dim, dec_hid_dim, dec_num_layers, drop_prob, num_mixture
        )

    def resampling(self, mu, std):
        z = torch.randn_like(mu).to(self.device)
        return mu + z * std

    def get_mixture_coef(self, output):
        z = output
        z_pen_logits = z[:, :, 0:3]  # pen states

        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = torch.chunk(z[:, :, 3:], 6, -1)

        # process output z's into MDN parameters

        # softmax all the pi's and pen states:
        z_pi = F.softmax(z_pi, dim=-1)
        z_pen = F.softmax(z_pen_logits, dim=-1)

        # exponentiate the sigmas and also make corr between -1 and 1.
        z_sigma1 = torch.exp(z_sigma1)
        z_sigma2 = torch.exp(z_sigma2)
        z_corr = torch.tanh(z_corr)

        return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen]

    def calc_reconstruction_loss(
        self, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, target
    ):
        x1, x2, eos, eoc, cont = torch.split(target, 1, dim=-1)

        # 1. 各成分の2次元ガウス分布の確率密度を計算
        norm1 = (x1 - z_mu1) / z_sigma1
        norm2 = (x2 - z_mu2) / z_sigma2
        z = norm1**2 + norm2**2 - 2 * z_corr * norm1 * norm2
        neg_rho = 1 - z_corr**2
        exp_part = torch.exp(-z / (2 * neg_rho))
        denom = 2 * np.pi * z_sigma1 * z_sigma2 * torch.sqrt(neg_rho)
        gaussian_prob = exp_part / denom

        # 2. 混合成分の確率で重み付けし、全体の確率密度を求める
        weighted_gaussian_prob = gaussian_prob * z_pi
        loss1 = -torch.log(
            weighted_gaussian_prob.sum(dim=-1) + 1e-5
        )  # log(0)回避のため小さな値を足す
        # 3. ペンの状態のクロスエントロピー損失
        pen_target = torch.cat([eos, eoc, cont], dim=-1)
        pen_loss = F.cross_entropy(
            z_pen.reshape(-1, 3), torch.argmax(pen_target, dim=2).reshape(-1)
        )

        # 再構築損失を求める
        reconstruction_loss = (loss1 * (1 - cont).squeeze()).mean() + pen_loss
        return reconstruction_loss

    def calc_kl_loss(self, mu, sigma):
        """KL損失の計算"""
        kl_loss = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
        return kl_loss

    def calculate_total_loss(self, output, target, mu, sigma, kl_weight=1.0):
        """Sketch-RNNのトータル損失の計算"""

        # GMMパラメータ取得
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen = output

        # KL損失の計算
        kl_loss = self.calc_kl_loss(mu, sigma)

        # 再構築損失の計算
        reconstruction_loss = self.calc_reconstruction_loss(
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, target
        )

        # 合計損失
        total_loss = reconstruction_loss + self.eta * kl_weight * torch.max(
            kl_loss, self.KL_min
        )
        # total_loss = reconstruction_loss + kl_weight * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def forward(self, x, max_len, seq_len):
        mu, std = self.encoder(x[:, : max_len + 1, :], seq_len)
        z = self.resampling(mu, std)
        z, _ = self.decoder(z, x[:, :-1, :])
        o = self.get_mixture_coef(z)
        return o, mu, std

    def sample_next_step(
        self, pi, mu1, mu2, sigma1, sigma2, corr, pen, temperature=1.0
    ):
        """次のステップをサンプリング"""

        # 1. 混合成分のインデックスをサンプリング
        pi = torch.log(pi) / temperature
        pi = torch.exp(pi - pi.max(dim=-1, keepdim=True).values)
        pi /= pi.sum(dim=-1, keepdim=True)  # 正規化
        pi_idx = torch.multinomial(pi, 1).item()  # サンプリングによるインデックス取得

        # 2. サンプリングするパラメータの選択
        mu1, mu2 = mu1[pi_idx], mu2[pi_idx]
        sigma1, sigma2 = sigma1[pi_idx], sigma2[pi_idx]
        corr = corr[pi_idx]

        # 3. 2次元ガウス分布からx, yをサンプリング
        mean = [mu1.item(), mu2.item()]
        cov = [
            [sigma1.item() ** 2, corr.item() * sigma1.item() * sigma2.item()],
            [corr.item() * sigma1.item() * sigma2.item(), sigma2.item() ** 2],
        ]
        x, y = np.random.multivariate_normal(mean, cov)

        # 4. ペンの状態をサンプリング
        pen = torch.multinomial(
            pen, 1
        ).item()  # 0: pen down, 1: pen up, 2: end of sketch

        return x, y, pen

    def sample(self, z, condition, seq_len=512, temperature=1.0):
        """サンプルシーケンスを生成"""
        strokes = condition

        prev_x = condition[-1].unsqueeze(0)
        output, state = self.decoder(z, condition.unsqueeze(0))

        for i in range(seq_len):
            # デコーダに入力
            output, state = self.decoder(z, prev_x, state=state, is_eval=True)

            # GMMパラメータ取得
            pi, mu1, mu2, sigma1, sigma2, corr, pen = self.get_mixture_coef(
                output.unsqueeze(0)
            )

            pi, mu1, mu2, sigma1, sigma2, corr, pen = (
                pi.squeeze(),
                mu1.squeeze(),
                mu2.squeeze(),
                sigma1.squeeze(),
                sigma2.squeeze(),
                corr.squeeze(),
                pen.squeeze(),
            )

            # サンプリング
            x, y, pen_state = self.sample_next_step(
                pi, mu1, mu2, sigma1, sigma2, corr, pen, temperature
            )

            # 次のステップに使用する入力の更新
            prev_x = torch.zeros(1, 5).to(z.device)
            prev_x[:, 0] = x
            prev_x[:, 1] = y
            prev_x[:, 2 + pen_state] = 1
            strokes = torch.concat([strokes, prev_x])

            # スケッチが終了した場合、ループを終了
            if pen_state == 2:
                break

        return strokes

    def cond_sampling(self, condition, seq_len=512, temperature=1.0):
        mu, std = self.encoder(condition.unsqueeze(0), is_eval=True)
        z = self.resampling(mu, std)
        return self.sample(z, condition, seq_len, temperature)
