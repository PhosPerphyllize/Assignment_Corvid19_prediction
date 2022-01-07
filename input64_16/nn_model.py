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

if __name__ == '__main__':
    nnline = nnLinear()
    nnconv = nnConv()
    input = torch.ones((1,64))
    print(input.shape)

    output = nnline(input)
    print(output.shape)

    output = nnconv(input)
    print(output.shape)