import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
from transformers import BertModel, BertTokenizer


class TextEmbeddingLayer(nn.Module):
    def __init__(self, pretrained_model, max_num_tweets, max_tokens, bert_dim=768, hidden_size=100):

        super(TextEmbeddingLayer, self).__init__()
        self.max_num_tweets = max_num_tweets
        self.max_tokens = max_tokens
        self.bert_dim = bert_dim
        self.hidden_size = hidden_size


        self.bert = BertModel.from_pretrained(pretrained_model)
        self.bert.requires_grad = True

        self.bi_gru = nn.GRU(
            input_size=bert_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.projection = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, text_inputs):
        """
        text_inputs: [batch_size, num_tweets, max_tokens]
        返回: [batch_size, num_tweets, hidden_size]
        """
        batch_size, num_tweets, max_tokens = text_inputs.shape
        text_embeddings = []

        for b in range(batch_size):
            tweet_embeds = []
            for t in range(num_tweets):
                tokens = text_inputs[b, t, :].unsqueeze(0)  # [1, max_tokens]
                with torch.no_grad():
                    outputs = self.bert(tokens)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, bert_dim]
                tweet_embeds.append(cls_embedding)


            tweet_embeds = torch.cat(tweet_embeds, dim=0).unsqueeze(0)  # [1, num_tweets, bert_dim]
            seq_lengths = torch.tensor([num_tweets] * batch_size)
            packed = pack_padded_sequence(tweet_embeds, seq_lengths, batch_first=True)
            outputs, _ = self.bi_gru(packed)
            unpacked, _ = pad_packed_sequence(outputs, batch_first=True)  # [1, num_tweets, 2*hidden_size]


            projected = self.projection(unpacked)  # [1, num_tweets, hidden_size]
            text_embeddings.append(projected.squeeze(0))  # [num_tweets, hidden_size]

        return torch.stack(text_embeddings, dim=0)  # [batch_size, num_tweets, hidden_size]

class TextSelectionUnit(nn.Module):
    def __init__(self, hidden_size):

        super(TextSelectionUnit, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, 1)
        self.b1 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, h_prev, text_embeddings):

        batch_size, num_tweets, _ = text_embeddings.shape
        h_prev_expanded = h_prev.unsqueeze(1).repeat(1, num_tweets, 1)  # [batch_size, num_tweets, hidden_size]


        attention_input = torch.tanh(self.W1(h_prev_expanded) + self.W2(text_embeddings) + self.b1)
        attention_scores = self.k(attention_input).squeeze(-1)  # [batch_size, num_tweets]
        vos = F.softmax(attention_scores, dim=1).unsqueeze(-1)

        # 加权选择文本嵌入
        selected_embedding = torch.sum(text_embeddings * vos, dim=1)  # [batch_size, hidden_size]
        return vos, selected_embedding


class TextMemoryUnit(nn.Module):
    def __init__(self, hidden_size):

        super(TextMemoryUnit, self).__init__()
        self.W3 = nn.Linear(2 * hidden_size, hidden_size)
        self.W4 = nn.Linear(2 * hidden_size, hidden_size)
        self.W5 = nn.Linear(2 * hidden_size, hidden_size)
        self.b3 = nn.Parameter(torch.zeros(hidden_size))
        self.b4 = nn.Parameter(torch.zeros(hidden_size))
        self.b5 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, selected_embedding, h_prev, l_prev):

        input_concat = torch.cat([selected_embedding, h_prev], dim=1)  # [batch_size, 2*hidden_size]


        f_t = torch.sigmoid(self.W3(input_concat) + self.b3)
        o_t = torch.sigmoid(self.W4(input_concat) + self.b4)
        l_t = torch.tanh(self.W5(input_concat) + self.b5)
        l_current = f_t * l_prev + o_t * l_t
        return l_current


class InformationFusionUnit(nn.Module):
    def __init__(self, hidden_size, price_dim=3):

        super(InformationFusionUnit, self).__init__()
        self.W6 = nn.Linear(hidden_size * 2 + price_dim, 1)
        self.W7 = nn.Linear(hidden_size * 2, hidden_size)
        self.W8 = nn.Linear(hidden_size + price_dim, hidden_size)

    def forward(self, price_data, l_current, h_prev):

        price_concat = torch.cat([price_data, l_current, h_prev], dim=1)  # [batch_size, price_dim+2*hidden_size]
        text_concat = torch.cat([l_current, h_prev], dim=1)  # [batch_size, 2*hidden_size]
        price_h_prev = torch.cat([price_data, h_prev], dim=1)  # [batch_size, price_dim+hidden_size]


        d_t = torch.sigmoid(self.W6(price_concat))
        h_l = torch.tanh(self.W7(text_concat))
        h_p = torch.tanh(self.W8(price_h_prev))
        h_current = d_t * h_p + (1 - d_t) * h_l
        return h_current


class SharedRepresentationLearning(nn.Module):
    def __init__(self, hidden_size, price_dim=3, num_days=5):

        super(SharedRepresentationLearning, self).__init__()
        self.hidden_size = hidden_size
        self.price_dim = price_dim
        self.num_days = num_days


        self.tsu = TextSelectionUnit(hidden_size)
        self.tmu = TextMemoryUnit(hidden_size)
        self.ifu = InformationFusionUnit(hidden_size, price_dim)


        self.h0 = nn.Parameter(torch.zeros(1, hidden_size))
        self.l0 = nn.Parameter(torch.zeros(1, hidden_size))
        nn.init.xavier_uniform_(self.h0)
        nn.init.xavier_uniform_(self.l0)

    def forward(self, text_embeddings, price_series):

        batch_size, num_days, num_tweets, _ = text_embeddings.shape
        h_prev = self.h0.repeat(batch_size, 1)  # [batch_size, hidden_size]
        l_prev = self.l0.repeat(batch_size, 1)  # [batch_size, hidden_size]
        vos_list = []


        for day in range(num_days):
            text_day = text_embeddings[:, day, :, :]  # [batch_size, num_tweets, hidden_size]
            price_day = price_series[:, day, :]  # [batch_size, price_dim]

            vos, selected_embedding = self.tsu(h_prev, text_day)
            l_current = self.tmu(selected_embedding, h_prev, l_prev)
            h_current = self.ifu(price_day, l_current, h_prev)


            vos_list.append(vos)
            h_prev, l_prev = h_current, l_current


        vos_list = torch.stack(vos_list, dim=1)  # [batch_size, num_days, num_tweets, 1]
        return h_prev, vos_list


class DeepRecurrentGeneration(nn.Module):
    def __init__(self, hidden_size, latent_dim=20, dropout=0.4):

        super(DeepRecurrentGeneration, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim


        self.enc_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )


        self.mu_post = nn.Linear(hidden_size * 2, latent_dim)
        self.log_sigma_post = nn.Linear(hidden_size * 2, latent_dim)


        self.mu_prior = nn.Linear(hidden_size * 2, latent_dim)
        self.log_sigma_prior = nn.Linear(hidden_size * 2, latent_dim)


        self.dec_gru = nn.GRU(
            input_size=hidden_size + latent_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )


        self.predict_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, log_sigma):

        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, h_series, y=None, training=True):

        batch_size, num_days, _ = h_series.shape
        kl_loss = 0.0
        z_post_list = []
        predictions = []
        enc_hidden = torch.zeros(1, batch_size, self.hidden_size).to(h_series.device)
        z_prev = torch.zeros(batch_size, self.latent_dim).to(h_series.device)

        for day in range(num_days):
            h = h_series[:, day, :]  # [batch_size, hidden_size]

            _, enc_hidden = self.enc_gru(h.unsqueeze(1), enc_hidden)
            enc_hidden = enc_hidden.squeeze(0)  # [batch_size, hidden_size]

            enc_input = torch.cat([enc_hidden, z_prev], dim=1)  # [batch_size, 2*hidden_size]

            if training:

                mu_post = self.mu_post(enc_input)
                log_sigma_post = self.log_sigma_post(enc_input)
                z_post = self.reparameterize(mu_post, log_sigma_post)
                z_post_list.append(z_post)


                mu_prior = self.mu_prior(enc_input)
                log_sigma_prior = self.log_sigma_prior(enc_input)


                kl = -0.5 * torch.sum(1 + log_sigma_post - mu_post.pow(2) - log_sigma_post.exp(), dim=1)
                kl_loss += kl.mean()
                z_prev = z_post
            else:

                mu_prior = self.mu_prior(enc_input)
                z_post = mu_prior
                z_post_list.append(z_post)
                z_prev = z_post

            dec_input = torch.cat([h, z_post], dim=1).unsqueeze(1)  # [batch_size, 1, hidden_size+latent_dim]
            dec_output, _ = self.dec_gru(dec_input)
            dec_output = self.dropout(dec_output.squeeze(1))  # [batch_size, hidden_size]

            pred = torch.sigmoid(self.predict_layer(dec_output))
            predictions.append(pred)


        predictions = torch.stack(predictions, dim=1)  # [batch_size, num_days, 1]
        z_post = torch.stack(z_post_list, dim=1)  # [batch_size, num_days, latent_dim]
        return predictions, z_post, kl_loss


class TemporalAttentionPrediction(nn.Module):
    def __init__(self, hidden_size):

        super(TemporalAttentionPrediction, self).__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, 1)
        self.final_layer = nn.Linear(hidden_size * 2, 1)

    def forward(self, dec_hidden, final_pred):

        batch_size, num_steps, _ = dec_hidden.shape

        q = final_pred.unsqueeze(1)  # [batch_size, 1, hidden_size]
        q_encoded = torch.tanh(self.W_q(dec_hidden))  # [batch_size, num_steps, hidden_size]
        k_encoded = torch.tanh(self.W_k(dec_hidden))
        k_score = self.w_k(k_encoded).squeeze(-1)  # [batch_size, num_steps]


        attention_scores = torch.matmul(q, q_encoded.transpose(1, 2))  # [batch_size, 1, num_steps]
        attention_weights = F.softmax(attention_scores, dim=2)  # [batch_size, 1, num_steps]


        value_vector = torch.matmul(attention_weights, dec_hidden).squeeze(1)  # [batch_size, hidden_size]
        fusion_vector = torch.cat([value_vector, final_pred], dim=1)  # [batch_size, 2*hidden_size]


        final_prediction = torch.sigmoid(self.final_layer(fusion_vector))  # [batch_size, 1]
        return final_prediction


class PEN(nn.Module):
    def __init__(self, pretrained_model, max_num_tweets=20, max_tokens=30, hidden_size=100, dropout=0.1, price_dim=5, latent_dim=20, num_days=5):
        """
        预测-解释网络（PEN）核心模型
        参数:
            pretrained_model: BERT预训练模型路径
            max_num_tweets: 单日最大文本数（对应论文中设为20）
            max_tokens: 单文本最大标记数（对应论文中设为30）
            hidden_size: 隐藏层维度
            price_dim: 价格特征维度（收盘价、最高价、最低价，共3维）
            num_days: 历史时间窗口天数（对应论文中5天滞后窗口）
            latent_dim: 潜在变量维度
            dropout:  dropout率
        """
        super(PEN, self).__init__()


        self.text_embedding = TextEmbeddingLayer(
            pretrained_model=pretrained_model,
            max_num_tweets=max_num_tweets,
            max_tokens=max_tokens,
            hidden_size=hidden_size
        )
        self.srl = SharedRepresentationLearning(
            hidden_size=hidden_size,
            price_dim=price_dim,
            num_days=num_days
        )
        self.drg = DeepRecurrentGeneration(
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            dropout=dropout
        )
        self.tap = TemporalAttentionPrediction(hidden_size=hidden_size)


        self.kl_regulator = nn.KLDivLoss(reduction='batchmean')
        self.num_days = num_days
        self.max_num_tweets = max_num_tweets

    def forward(self, text_inputs, price_series, labels=None, training=True):
        """
        前向传播流程
        参数:
            text_inputs: [batch_size, num_days, num_tweets, max_tokens]
            price_series: [batch_size, num_days, price_dim]
            labels: [batch_size, num_days] (训练时使用)
            training: 是否为训练模式
        返回:
            在训练模式下返回 (final_pred, kl_loss, vos_list, drg_predictions)
            在推理模式下返回 final_pred
        """
        batch_size = text_inputs.shape[0]

        # 1. TEXT_Embedding
        batch_text_embeddings = []
        for day in range(self.num_days):
            day_text_inputs = text_inputs[:, day, :, :]  # [batch_size, num_tweets, max_tokens]
            day_embeddings = self.text_embedding(day_text_inputs)  # [batch_size, num_tweets, hidden_size]
            batch_text_embeddings.append(day_embeddings)

        text_embeddings = torch.stack(batch_text_embeddings, dim=1)  # [batch_size, num_days, num_tweets, hidden_size]

        # 2. SRL
        final_h, vos_list = self.srl(text_embeddings, price_series)  # final_h: [batch_size, hidden_size]

        # 3. DRG
        drg_predictions, z_post, kl_drg = self.drg(
            h_series=text_embeddings.reshape(batch_size, self.num_days, -1),
            y=labels,
            training=training
        )  # drg_predictions: [batch_size, num_days, 1]

        # 4. TAP
        dec_hidden = drg_predictions[:, :-1, :].reshape(batch_size, self.num_days - 1,
                                                        -1)  # [batch_size, num_days-1, hidden_size]
        final_pred = self.tap(dec_hidden, final_h)  # [batch_size, 1]

        if training:

            uniform_dist = torch.ones_like(vos_list) / self.max_num_tweets
            kl_reg = self.kl_regulator(vos_list.log(), uniform_dist)


            kl_loss = kl_drg + kl_reg
            return final_pred, kl_loss, vos_list, drg_predictions
        else:
            return final_pred

    def calculate_explainability(self, vos_list):
        """计算可解释性指标（如RTT）"""
        # 计算前两大VoS权重之和占比
        sorted_vos, _ = torch.sort(vos_list.squeeze(-1), dim=2, descending=True)
        top_two_sum = sorted_vos[:, :, 0] + sorted_vos[:, :, 1]
        rtt = (top_two_sum > 0.95).float().mean()
        return rtt
