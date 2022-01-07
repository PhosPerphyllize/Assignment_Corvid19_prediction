import torch
from torch import nn

class nnLinear(nn.Module):
    def __init__(self):
        super(nnLinear, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(671, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 30),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class nnConv(nn.Module):
    def __init__(self):
        super(nnConv, self).__init__()
        self.model = nn.Sequential(
            nn.Unflatten(1, (1,1,671)),
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(671,128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 30),
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    nnline = nnLinear()
    nnconv = nnConv()
    input = torch.ones((16, 671))
    print("linear:",end=" ")
    output = nnline(input)
    print(output.shape)

    print("conv:",end=" ")
    output = nnconv(input)
    print(output.shape)