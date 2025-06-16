import torch
import torch.nn as nn
import torch.nn.functional as F

class ALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_first, attention_size):
        super(ALSTM, self).__init__()

        self.hidden_size = hidden_size


        self.encoder_rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.encoder_bn = nn.BatchNorm1d(hidden_size)


        self.decoder_rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.decoder_bn = nn.BatchNorm1d(hidden_size)


        self.attention = nn.Linear(hidden_size * 2, attention_size)
        self.attention_score = nn.Linear(attention_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def attention_layer(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  # (batch_size, seq_len, hidden_dim)

        energy = torch.tanh(self.attention(torch.cat((decoder_hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, attention_dim)
        attn_weights = F.softmax(self.attention_score(energy).squeeze(2), dim=1)  # (batch_size, seq_len)


        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)

        return context, attn_weights

    def forward(self, x):
        encoder_outputs, (hidden, cell) = self.encoder_rnn(x)  # 编码器
        encoder_outputs = self.encoder_bn(encoder_outputs.transpose(1, 2)).transpose(1, 2)  # 批量归一化

        decoder_hidden, decoder_cell = hidden, cell

        context, attn_weights = self.attention_layer(decoder_hidden[-1], encoder_outputs)

        decoder_input = context.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        decoder_output, _ = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))
        decoder_output = self.decoder_bn(decoder_output.transpose(1, 2)).transpose(1, 2)  # 批量归一化

        decoder_output = self.dropout(decoder_output)
        predictions = self.fc_out(decoder_output.squeeze(1))

        predictions = self.sigmoid(predictions)

        return predictions
