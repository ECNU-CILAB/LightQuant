import torch
import torch.nn as nn
import torch.nn.functional as F

class ALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first):
        super(ALSTM, self).__init__()
        self.hidden_size = hidden_size

        self.encoder_rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)


        self.decoder_rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)


        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def attention_layer(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  # (batch_size, seq_len, hidden_dim)


        energy = torch.tanh(self.attention(torch.cat((decoder_hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(energy.sum(dim=2), dim=1)  # (batch_size, seq_len)


        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)

        return context, attn_weights

    def forward(self, x, output_size):
        # 编码器部分
        encoder_outputs, (hidden, cell) = self.encoder_rnn(x)  # encoder_outputs: (batch_size, seq_len, hidden_dim)

        # 解码器部分，只使用最后一个时间步的隐藏状态作为初始输入
        decoder_hidden, decoder_cell = hidden, cell

        # 使用Attention机制计算上下文向量
        context, attn_weights = self.attention_layer(decoder_hidden[-1], encoder_outputs)  # (batch_size, hidden_dim)

        # 解码器处理上下文向量
        decoder_input = context.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        decoder_output, _ = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))  # (batch_size, 1, hidden_dim)

        # Dropout层
        decoder_output = self.dropout(decoder_output)

        # 输出层生成最终预测
        fc_out = nn.Linear(self.hidden_size, output_size).to(x.device)
        predictions = fc_out(decoder_output.squeeze(1))  # (batch_size, output_dim)

        return predictions, attn_weights

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first):
        super(Discriminator, self).__init__()
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        if self.batch_first:
            out = out[:, -1, :]
        else:
            out = out[-1, :, :]
        out = self.fc_out(out)
        out = self.sigmoid(out)
        return out

class AdvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first):
        super(AdvLSTM, self).__init__()
        self.generator = ALSTM(input_size, hidden_size, num_layers, dropout, batch_first)
        self.discriminator = Discriminator(input_size, hidden_size, num_layers, dropout, batch_first)


    def generate(self, x=None, output_size=2):
        return self.generator(x, output_size)

    def discriminate(self, x):
        return self.discriminator(x)
