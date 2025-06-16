import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification



class HAN(nn.Module):
    def __init__(self, hidden_size, bert_dim, pretrained_model, days, max_num_tweets_len, dropout):
        super(HAN, self).__init__()

        self.bertmodel = AutoModelForSequenceClassification.from_pretrained(pretrained_model).base_model
        self.embedding_dim = bert_dim
        self.gru_dim = bert_dim
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.days = days
        self.max_num_tweets_len = max_num_tweets_len

        self.bertmodel.requires_grad_(False)
        # self.bertmodel.base_model.encoder.layer[-1].requires_grad_(True) #fine tune only last encoder layer
        # self.bertmodel.base_model.requires_grad_(True)

        # data_shape (batchsize, days, max_daily_news, embedding_dim=756)

        self.bi_gru = nn.GRU(self.embedding_dim, self.gru_dim, bidirectional=True, batch_first=True)
        self.attn0 = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid(),
            nn.Softmax(dim=2),
        )
        self.attn1 = nn.Sequential(
            nn.Linear(2 * self.gru_dim, 1),
            nn.Sigmoid(),
            nn.Softmax(dim=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * self.gru_dim, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 4, 1)
        )

        # self.fc0 = nn.Linear(2 * self.gru_dim, self.hidden_size)
        # self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        # self.elu = nn.ELU()
        # self.fc_out = nn.Linear(self.hidden_size, 2)

        # nn.init.xavier_normal_(self.attn0[0].weight)
        # nn.init.xavier_normal_(self.attn1[0].weight)

    def forward(self, input):
        # input: (batch_size=32, days=5, max_daily_news=40, sentence_type=3, max_tokens=30)
        x = torch.zeros(input.shape[:3] + (self.embedding_dim,))

        # get embeddings
        for i in range(self.days):
            for j in range(self.max_num_tweets_len):
                x[:, i, j, :] = self.bertmodel(
                    input_ids=input[:, i, j, 0, :],
                    token_type_ids=input[:, i, j, 1, :],
                    attention_mask=input[:, i, j, 2, :],
                ).last_hidden_state[:, 0, :]  # hidden CLS token for sentence embedding (batch_size, tokens, 768)
                # .max(dim=1)[0] # max_pooling
                # .mean(dim=1) # average_pooling

        # x_shape: (batch_size=32, days=5, max_daily_news=40, embedding_dim=768)
        # news level attention
        x = torch.sum((self.attn0(x)) * x, dim=2)  # (32, 5, 768)
        # bi_gru
        x = self.bi_gru(x)[0]  # (32, 5, 768*2)
        # temporal attention
        x = torch.sum((self.attn1(x)) * x, dim=1)  # (32, 768*2)
        x = self.fc(x)
        return x


# class BERTforFinetuning(nn.Module):
#     def __init__(self, flags):
#         super(BERTforFinetuning, self).__init__()
#
#         self.flags = flags
#         self.bertmodel = AutoModelForSequenceClassification.from_pretrained(flags.modelpath).base_model
#         self.embedding_dim = 768
#         self.num_class = flags.num_class
#
#         self.classifier = nn.Linear(self.embedding_dim, self.num_class)
#
#     def forward(self, input):
#         # input (batch_size, id_type=3, max_tokens=30)
#         m = self.bertmodel(
#             input_ids=input[:, 0, :],
#             token_type_ids=input[:, 1, :],
#             attention_mask=input[:, 2, :],
#         ).last_hidden_states[:, 0, :]  # (batch_size, tokens, 768)
#
#         y = self.classifier(m)
#         return y
