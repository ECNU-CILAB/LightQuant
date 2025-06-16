import torch
import torch.nn as nn
from utils.news_process import NewsEmbeddingModel

class MarketInformationEncoder(nn.Module):
    def __init__(self, market_dim, news_dim, hidden_dim):
        super(MarketInformationEncoder, self).__init__()
        self.market_gru = nn.GRU(market_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.news_gru = nn.GRU(news_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.final_fc = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, market_x, news_x):
        """
        :param market_x: (batch_size, look_back_window, market_dim)
        :param news_x: (batch_size, look_back_window, news_dim)
        """

        market_h, _ = self.market_gru(market_x)  # (batch, look_back_window, hidden_dim*2)
        market_attn_weights = torch.softmax(self.attn(market_h), dim=1)
        market_context = torch.sum(market_h * market_attn_weights, dim=1)  # (batch, hidden_dim*2)

        news_h, _ = self.news_gru(news_x)  # (batch, look_back_window, hidden_dim*2)
        news_attn_weights = torch.softmax(self.attn(news_h), dim=1)
        news_context = torch.sum(news_h * news_attn_weights, dim=1)  # (batch, hidden_dim*2)

        # Fuse Market and News Information
        final_context = torch.relu(self.final_fc(torch.cat((market_context, news_context), dim=-1)))  # (batch, hidden_dim)
        return final_context


# Variational Movement Decoder (VMD)
class VariationalMovementDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalMovementDecoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, 2)

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


class AttentiveTemporalAuxiliary(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentiveTemporalAuxiliary, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, history_outputs):

        h = torch.tanh(self.fc(history_outputs))  # (batch_size, look_back_window, hidden_dim)
        attn_weights = torch.softmax(self.attention(h), dim=1)  # (batch_size, look_back_window, 1)
        context = torch.sum(h * attn_weights, dim=1)  # (batch_size, hidden_dim)
        return context

# StockNet
class StockNet(nn.Module):
    def __init__(self, market_dim, news_dim, hidden_dim, latent_dim, look_back_window):
        super(StockNet, self).__init__()
        self.encoder = MarketInformationEncoder(market_dim, news_dim, hidden_dim)
        self.decoder = VariationalMovementDecoder(hidden_dim, hidden_dim, latent_dim)
        self.temporal_attn = AttentiveTemporalAuxiliary(hidden_dim)
        self.look_back_window = look_back_window
        self.attn_fc = nn.Linear(hidden_dim, 2)
        self.prediction_fc = nn.Linear(2, hidden_dim)
        # news embedding model
        self.news_embedding = NewsEmbeddingModel(
            encoder_type="gru",
            input_dim=768,
            hidden_dim=256,
            output_dim=news_dim
        )

    def forward(self, market_x, news_x, history_outputs):

        news_x = self.news_embedding(news_x)
        context = self.encoder(market_x, news_x)
        output, mu, logvar = self.decoder(context.unsqueeze(1))

        # Temporal Attention Enhancement
        attn_context_out = self.temporal_attn(history_outputs)
        attn_context = self.attn_fc(attn_context_out)

        prediction = torch.softmax(output + attn_context, dim=-1)

        prediction_mapped = self.prediction_fc(prediction)

        return prediction, mu, logvar, prediction_mapped

