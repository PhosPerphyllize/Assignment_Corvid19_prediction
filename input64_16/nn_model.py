import torch
from torch import nn

class nnLinear(nn.Module):
    def __init__(self):
        super(nnLinear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(256, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class nnConv(nn.Module):
    def __init__(self):
        super(nnConv, self).__init__()
        self.model = nn.Sequential(
            nn.Unflatten(1, (1,1,64)),
            nn.Conv2d(in_channels=1,out_channels=4,kernel_size=5,padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256,64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class nnRNN(nn.Module):
    def __init__(self):
        super(nnRNN, self).__init__()
        input_size = 1
        hidden_size = 16
        num_layers = 4
        self.rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
                          nonlinearity='relu',batch_first=True)
        linear_in = hidden_size*64
        self.linear = nn.Sequential(
            nn.Linear(linear_in, 128),  # 1024-128
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        # x (batch_size, input_size)
        # rnn_in (batch_size, seq_len, input_size)
        # out (batch_size, hidden_size)
        b, _ = x.size()
        x = x.reshape(b,64,-1)   # 要使用batch_first=True
        x, _ = self.rnn(x)
        b, s, n = x.size()
        x = x.reshape(b, -1)
        x = self.linear(x)
        return x

class nnLSTM(nn.Module):
    def __init__(self):
        super(nnLSTM, self).__init__()
        input_size = 1
        hidden_size = 16
        num_layers = 4
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        linear_in = hidden_size*64
        self.linear = nn.Sequential(
            nn.Linear(linear_in, 128),  # 1024-128
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        # x (batch_size, input_size)
        # rnn_in (batch_size, seq_len, input_size)
        # out (batch_size, hidden_size)
        b, _ = x.size()
        x = x.reshape(b, 64, -1)  # 要使用batch_first=True
        x, _ = self.lstm(x)
        b,s,n = x.size()
        x = x.reshape(b,-1)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    nnline = nnLinear()
    nnconv = nnConv()
    nnrnn = nnRNN()
    nnlstm = nnLSTM()
    input = torch.rand((2,64))  # batch_size, input_size
    print(input)
    print(input.shape)

    output = nnline(input)
    print(output.shape)

    output = nnconv(input)
    print(output.shape)

    print("rnn: ")
    output = nnrnn(input)
    print(output.shape)
    print(output)

    print("lstm: ")
    output = nnlstm(input)
    print(output.shape)
    print(output)