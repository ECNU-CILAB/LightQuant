import torch
import torch.nn as nn
import torch.nn.functional as F

class ALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_first):
        super(ALSTM, self).__init__()
        # 编码器
        self.encoder_rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)

        # 解码器
        self.decoder_rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)

        # Attention机制
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def attention_layer(self, decoder_hidden, encoder_outputs):
        """
        计算注意力权重并生成上下文向量
        Args:
            decoder_hidden: 解码器的隐藏状态 (batch_size, hidden_dim)
            encoder_outputs: 编码器的所有输出 (batch_size, seq_len, hidden_dim)
        Returns:
            context: 上下文向量 (batch_size, hidden_dim)
            attn_weights: 注意力权重 (batch_size, seq_len)
        """

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  # (batch_size, seq_len, hidden_dim)

        # 计算注意力分数
        energy = torch.tanh(self.attention(torch.cat((decoder_hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(energy.sum(dim=2), dim=1)  # (batch_size, seq_len)

        # 加权求和编码器输出，得到上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)

        return context, attn_weights

    def forward(self, x):
        """
        前向传播：编码、注意力机制和解码
        Args:
            x: 输入数据 (batch_size, seq_len, input_dim)
        Returns:
            predictions: 预测结果 (batch_size, output_dim)
            attn_weights: 注意力权重 (batch_size, seq_len)
        """
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
        predictions = self.fc_out(decoder_output.squeeze(1))  # (batch_size, output_dim)

        return predictions
