# author:Liu Yu
# time:2025/3/12 16:57
import torch
import torch.nn as nn

class MarketInformationEncoder(nn.Module):
    def __init__(self, market_dim, news_dim, hidden_dim):
        super(MarketInformationEncoder, self).__init__()
        self.market_gru = nn.GRU(market_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.news_gru = nn.GRU(news_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.final_fc = nn.Linear(hidden_dim * 4, hidden_dim)  # 融合市场和新闻

    def forward(self, market_x, news_x):
        """
        :param market_x: (batch_size, look_back_window, market_dim)
        :param news_x: (batch_size, look_back_window, news_dim)
        """
        # 处理市场数据
        market_h, _ = self.market_gru(market_x)  # (batch, look_back_window, hidden_dim*2)
        market_attn_weights = torch.softmax(self.attn(market_h), dim=1)
        market_context = torch.sum(market_h * market_attn_weights, dim=1)  # (batch, hidden_dim*2)

        # 处理新闻数据
        news_h, _ = self.news_gru(news_x)  # (batch, look_back_window, hidden_dim*2)
        news_attn_weights = torch.softmax(self.attn(news_h), dim=1)
        news_context = torch.sum(news_h * news_attn_weights, dim=1)  # (batch, hidden_dim*2)

        # 融合市场和新闻信息
        final_context = torch.relu(self.final_fc(torch.cat((market_context, news_context), dim=-1)))  # (batch, hidden_dim)
        return final_context

# 变分运动解码器（VMD）
class VariationalMovementDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalMovementDecoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, 2)  # 输出二分类：涨 or 跌

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h, _ = self.gru(x)
        mu, logvar = self.fc_mu(h[:, -1, :]), self.fc_logvar(h[:, -1, :])
        z = self.reparameterize(mu, logvar)
        output = self.fc_out(z)
        return output, mu, logvar

# 时间注意力机制（ATA）
class AttentiveTemporalAuxiliary(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentiveTemporalAuxiliary, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, history_outputs):
        """
        :param history_outputs: (batch_size, look_back_window, hidden_dim)
        """
        h = torch.tanh(self.fc(history_outputs))  # (batch_size, look_back_window, hidden_dim)
        attn_weights = torch.softmax(self.attention(h), dim=1)  # (batch_size, look_back_window, 1)
        context = torch.sum(h * attn_weights, dim=1)  # (batch_size, hidden_dim)
        return context

# StockNet 总模型
class StockNet(nn.Module):
    def __init__(self, market_dim, news_dim, hidden_dim, latent_dim, look_back_window):
        super(StockNet, self).__init__()
        self.encoder = MarketInformationEncoder(market_dim, news_dim, hidden_dim)
        self.decoder = VariationalMovementDecoder(hidden_dim, hidden_dim, latent_dim)
        self.temporal_attn = AttentiveTemporalAuxiliary(hidden_dim)
        self.look_back_window = look_back_window
        self.attn_fc = nn.Linear(hidden_dim, 2)
        self.prediction_fc = nn.Linear(2, hidden_dim)
    def forward(self, market_x, news_x, history_outputs):
        context = self.encoder(market_x, news_x)
        output, mu, logvar = self.decoder(context.unsqueeze(1))

        # 时间注意力增强
        attn_context_out = self.temporal_attn(history_outputs)
        attn_context = self.attn_fc(attn_context_out)

        prediction = torch.softmax(output + attn_context, dim=-1)  # 融合历史预测

        prediction_mapped = self.prediction_fc(prediction)

        return prediction, mu, logvar, prediction_mapped

