import torch
from torch import nn

class nnLinear(nn.Module):
    def __init__(self):
        super(nnLinear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256,256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 16),
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
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=16,  # RNN隐藏神经元个数
            num_layers=1,  # RNN隐藏层个数
            nonlinearity='relu',
            batch_first=True
        )

    def forward(self, x):
        # x (time_step, batch_size, input_size)
        # h (n_layers, batch, hidden_size)
        # out (time_step, batch_size, hidden_size)
        output, h_n = self.rnn(x)
        return output

class nnLSTM(nn.Module):
    def __init__(self):
        super(nnLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=16,  # 隐藏神经元个数
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # x (time_step, batch_size, input_size)
        # h (n_layers, batch, hidden_size)
        # out (time_step, batch_size, hidden_size)
        output, hc_n = self.lstm(x)
        return output

if __name__ == '__main__':
    nnline = nnLinear()
    nnconv = nnConv()
    nnrnn = nnRNN()
    nnlstm = nnLSTM()
    input = torch.rand((8,64))  # batch_size, input_size
    # print(input)
    print(input.shape)

    output = nnline(input)
    print(output.shape)

    output = nnconv(input)
    print(output.shape)

    input = input.view(8,64,-1)   # 要使用batch_first=True
    print(input.shape)
    output = nnrnn(input)
    output = output[:, -1, :]
    print(output.shape)  # 取出最后一个，其形式为batch_size * output
    # print(output)

    output = nnlstm(input)
    output = output[:, -1, :]
    print(output.shape)  # 取出最后一个，其形式为batch_size * output
    # print(output)

