import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class AttentiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_size, perturbation_size, epsilon):
        super(AttentiveLSTM, self).__init__()

        # Feature mapping layer (fully connected layer)
        self.fc = nn.Linear(input_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Temporal attention layer
        self.attention_fc = nn.Linear(hidden_size, attention_size)
        self.attention_weight = nn.Parameter(torch.randn(attention_size))

        # Prediction layer
        self.predict_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size),
            nn.Sigmoid()
        )

        # Hyperparameters for adversarial training
        self.perturbation_size = perturbation_size
        self.epsilon = epsilon

    def attention(self, lstm_output):
        # Temporal Attention mechanism
        attention_scores = torch.matmul(lstm_output, self.attention_weight)  # Calculate attention score
        attention_weights = F.softmax(attention_scores, dim=1)
        attended_output = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)  # Aggregate the output
        return attended_output

    def forward(self, x):
        # Feature mapping
        mapped_input = torch.tanh(self.fc(x))

        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(mapped_input)

        # Attention mechanism
        attended_output = self.attention(lstm_out)

        # Concatenate attention output and last hidden state
        final_representation = torch.cat((attended_output, hn[-1]), dim=-1)

        # Prediction layer
        prediction = self.predict_fc(final_representation)
        return prediction


class AdvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_size, perturbation_size, epsilon):
        super(AdvLSTM, self).__init__()
        self.attentive_lstm = AttentiveLSTM(input_size, hidden_size, output_size, attention_size, perturbation_size,
                                            epsilon)
        self.perturbation_size = perturbation_size
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = epsilon

    def adversarial_loss(self, clean_output, adv_output, labels):
        # Adversarial loss function
        clean_loss = self.criterion(clean_output, labels)
        adv_loss = self.criterion(adv_output, labels)
        return clean_loss, adv_loss

    def generate_adversarial_example(self, x, clean_output, labels):
        if not x.requires_grad:
            x.requires_grad_()  # Ensure that the input tensor x has gradient computation enabled.

        # 计算对抗损失
        loss = self.criterion(clean_output, labels)
        loss.backward(retain_graph=True)  # compute gradients

        # 获取梯度方向上的扰动
        grad = x.grad.data
        adversarial_perturbation = self.epsilon * grad / grad.norm()  # Normalize the perturbation.

        # Add an additional random noise perturbation.
        random_perturbation = (torch.rand_like(
            x) - 0.5) * 2 * self.perturbation_size  # [-perturbation_size, perturbation_size]


        adversarial_example = x + adversarial_perturbation + random_perturbation
        adversarial_example = torch.clamp(adversarial_example, min=0, max=1)

        return adversarial_example

    def forward(self, x, labels=None):
        # Get the prediction for the clean (original) input sample.
        clean_output = self.attentive_lstm(x)
        clean_output = clean_output.squeeze()


        if self.training and labels is not None:
            x.requires_grad_()  # Ensure gradients are computed for the input
            adv_example = self.generate_adversarial_example(x, clean_output, labels)
            adv_output = self.attentive_lstm(adv_example)


            clean_loss, adv_loss = self.adversarial_loss(clean_output, adv_output, labels)


            total_loss = clean_loss + 0.1*adv_loss
            return total_loss

        return clean_output


