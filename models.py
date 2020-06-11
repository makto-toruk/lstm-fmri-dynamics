import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
'''
classifier (LSTM)
k_feat: k_hidden: k_class
'''
class LSTMClassifier(nn.Module):
    def __init__(self, k_input, k_hidden, k_layers, 
        k_class, return_states=False):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(k_input, k_hidden,
            k_layers, batch_first=True)
        self.fc = nn.Linear(k_hidden, k_class)
        self.return_states = return_states

    def forward(self, x, x_len, max_length=None):
        x = pack_padded_sequence(x, x_len, 
            batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x) # h0, c0 initialized to zero; 
            # ignore last hidden state
        x, _ = pad_packed_sequence(x, batch_first=True, 
            total_length=max_length) # ignore length
        y = self.fc(x)

        if self.return_states:
            return x, y
        else:
            return y
'''
encoder (LSTM)
k_feat: k_hidden: k_dim: k_class
trained as a classifier
'''
class LSTMEncoder(nn.Module):
    def __init__(self, 
        k_input, k_hidden, k_dim, k_layers,
        k_class, return_states=False):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(k_input, k_hidden,
            k_layers, batch_first=True)
        self.fc1 = nn.Linear(k_hidden, k_dim)
        self.fc2 = nn.Linear(k_dim, k_class) 
        self.return_states = return_states

    def forward(self, x, x_len, max_length=None):
        x = pack_padded_sequence(x, x_len, 
            batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x) # h0, c0 initialized to zero; 
            # ignore last hidden state
        x, _ = pad_packed_sequence(x, batch_first=True, 
            total_length=max_length) # ignore length
        x = self.fc1(x)
        y = self.fc2(x)

        if self.return_states:
            return x, y
        else:
            return y
'''
decoder (LSTM)
k_input: k_hidden
trained for reconstruction
'''
class LSTMDecoder(nn.Module):
    def __init__(self, k_input, k_hidden, k_layers):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(k_input, k_hidden,
            k_layers, batch_first=True)
        
    def forward(self, x, x_len, max_length=None):
        x = pack_padded_sequence(x, x_len, 
            batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x) # h0, c0 initialized to zero; 
            # ignore last hidden state
        x, _ = pad_packed_sequence(x, batch_first=True, 
            total_length=max_length) # ignore length
        return x
'''
classifier (FeedForward)
k_feat: (k_hidden:)*k_layers: k_class
'''
class FFClassifier(nn.Module):
    def __init__(self, k_input, k_hidden, k_layers, 
        k_class, return_states=False):
        super(FFClassifier, self).__init__()
        self.k_layers = k_layers
        self.return_states = return_states
        self.fc = nn.ModuleList([])
        for ii in range(k_layers):
            if ii==0:
                self.fc.append(nn.Linear(k_input, k_hidden))
            elif ii==k_layers-1:
                self.fc.append(nn.Linear(k_hidden, k_class))
            else:
                self.fc.append(nn.Linear(k_hidden, k_hidden))

    def forward(self, x):
        for ii in range(self.k_layers-1):
            x = F.relu(self.fc[ii](x))
        y = self.fc[ii+1](x)
        
        if self.return_states:
            return x, y
        else:
            return y

'''
classifier (TemporalCNN)
k_feat: k_hidden: k_class
'''
class TCNClassifier(nn.Module):
    def __init__(self, k_input, k_hidden, k_wind, 
        k_class, return_states=False):
        super(TCNClassifier, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,
            out_channels=k_hidden,
            kernel_size=(k_input, k_wind),
            stride=1)
        # left pad zeros
        self.pad = nn.ZeroPad2d((k_wind-1, 0, 0, 0))
        self.fc = nn.Linear(k_hidden, k_class)
        self.return_states = return_states

    def forward(self, x):
        # x: (k_seq x time x k_input)
        # x.permute: (k_seq x k_input x time)
        # unsqueeze: specifies 1 channel
        x = F.relu(self.conv(
            self.pad(x.permute(0, 2, 1).unsqueeze(1))))
        # squeeze: H = 1 based on kernel size
        # x.permute: (k_seq x time x k_hidden)
        y = self.fc(x.squeeze(2).permute(0, 2, 1))

        if self.return_states:
            return x, y
        else:
            return y
'''
regression (LSTM)
k_feat: k_hidden: 1
'''
class LSTMRegression(nn.Module):
    def __init__(self, k_input, k_hidden, k_layers, 
        return_states=False):
        super(LSTMRegression, self).__init__()
        self.lstm = nn.LSTM(k_input, k_hidden,
            k_layers, batch_first=True)
        self.fc = nn.Linear(k_hidden, 1)
        self.return_states = return_states

    def forward(self, x, x_len, max_length=None):
        x = pack_padded_sequence(x, x_len, 
            batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x) # h0, c0 initialized to zero; 
            # ignore last hidden state
        x, _ = pad_packed_sequence(x, batch_first=True, 
            total_length=max_length) # ignore length
        y = self.fc(x)

        if self.return_states:
            return x, y
        else:
            return y

class LSTMEncReg(nn.Module):
    def __init__(self, k_input, k_hidden, k_dim,
        k_layers, return_states=False):
        super(LSTMEncReg, self).__init__()
        self.lstm = nn.LSTM(k_input, k_hidden,
            k_layers, batch_first=True)
        self.fc1 = nn.Linear(k_hidden, k_dim)
        self.fc2 = nn.Linear(k_dim, 1) 
        self.return_states = return_states

    def forward(self, x, x_len, max_length=None):
        x = pack_padded_sequence(x, x_len, 
            batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x) # h0, c0 initialized to zero; 
            # ignore last hidden state
        x, _ = pad_packed_sequence(x, batch_first=True, 
            total_length=max_length) # ignore length
        x = self.fc1(x)
        y = self.fc2(x)

        if self.return_states:
            return x, y
        else:
            return y

'''
classifier (LogReg)
k_feat: k_class
'''
class LogReg(nn.Module):
    def __init__(self, k_input, k_class):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(k_input, k_class)

    def forward(self, x):
        y = self.fc(x)

        return y
