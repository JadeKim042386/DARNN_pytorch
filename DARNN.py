import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in DA_RNN.
        
        Args:
            T: time step
        """
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            batch_first=True
        )

        self.we = nn.Linear(in_features=2 * self.encoder_num_hidden, out_features=self.T - 1)
        self.ue = nn.Linear(in_features=self.T - 1, out_features=self.T - 1)
        self.ve = nn.Linear(in_features=self.T - 1, out_features=1)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data
        """
        X_tilde = Variable(X.data.new(X.size(0), self.T - 1, self.input_size).zero_())

        # h_n, s_n: initial states with dimention hidden_size
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        for t in range(self.T - 1):
            x = X[:, t, :]
            x = x[:, np.newaxis, :]

            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # Eq. 2
            _, final_state = self.encoder_lstm(x, (h_n, s_n))
            h_s, c_s = final_state[0], final_state[1]

            # [h_{t-1}; s_{t-1}]
            query = torch.cat((h_s.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                               c_s.repeat(self.input_size, 1, 1).permute(1, 0, 2)), dim=2)

            # x^k
            x_perm = X.permute(0, 2, 1)

            # Eq. 8: v_e^T tanh(W_e[h_{t-1}; s_{t-1}] + U_e * x^k)
            score = self.ve(torch.tanh(self.we(query) + self.ue(x_perm)))
            score = score.permute(0, 2, 1)

            # Eq. 9: Attention weights
            attention_weights = F.softmax(score.view(-1, self.input_size), dim=1)
            
            # Eq. 10: New input at time t
            x_tilde = torch.mul(attention_weights, X[:, t, :])
            X_tilde[:, t, :] = x_tilde

        # Eq. 11: Update encoder hidden state
        enc_h, _ = self.lstm(X_tilde)
        return enc_h

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X: input data

        Returns:
            initial_states
        """
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_RNN.
        
        Args:
            T: time step
        """
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.vd = nn.Linear(in_features=self.decoder_num_hidden, out_features=1)
        self.wd = nn.Linear(in_features=2 * self.decoder_num_hidden, out_features=self.encoder_num_hidden)
        self.ud = nn.Linear(in_features=self.encoder_num_hidden, out_features=self.encoder_num_hidden)

        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )
        
        self.fc = nn.Linear(encoder_num_hidden + 1, 1) # Eq. 15: w_tilde
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, decoder_num_hidden) # Eq. 22: W_y
        self.vy = nn.Linear(decoder_num_hidden, 1) # Eq. 22: v_y

        self.fc.weight.data.normal_()

    def forward(self, enc_h, y_prev):
        """forward.

        Args:
            enc_h: encoder hidden state
            y_prev: decoder previous cell state
        """
        d_n = self._init_states(enc_h)
        c_n = self._init_states(enc_h)

        for t in range(self.T - 1):
            # [d_{t-1}; s'_{t-1}]
            query = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                                c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2)), dim=2)

            # Eq. 12: v_d^T tanh(W_d[d_{t-1}; s'_{t-1}] + U_d * h_i)
            score = self.vd(torch.tanh(self.wd(query) + self.ud(enc_h)))
            score = score.permute(0, 2, 1)

            # Eq. 13: Attention weights
            beta = F.softmax(score, dim=2)

            # Eq. 14: Compute context vector
            context = torch.bmm(beta, enc_h)[:, 0, :]
            if t < self.T - 1:
                # Eq. 15: New decoder input
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))

                # Eq. 16: Update decoder hidden state
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]
                c_n = final_states[1]

        # Eq. 22: Final prediction
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))
        y_pred = self.vy(y_pred)

        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X: input data
            
        Returns:
            initial_states
        """
        # hidden state and cell state
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())
