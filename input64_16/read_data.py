
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):
    def __init__(self, root:str, train:bool=True):
        self.train = train
        self.tab_main = np.genfromtxt(root, delimiter=",")

        ratio = 0.01  # 测试集与训练集比
        self.output_day = len(self.tab_main[0,:])  # 用作输出的天数的序号
        self.input_day = self.output_day - 16  # 输出天数为16

        list_num = len(self.tab_main[:,0])   # 输入表的总样本数
        testset_num = int(ratio*list_num)
        trainset_num = list_num-testset_num
        if self.train:
            self.list = np.arange(0, trainset_num)   # 训练集行号
        else:
            self.list = np.arange(trainset_num, list_num)  # 测试集行号

    def __getitem__(self, idx):
        tab_idx = self.list[idx]
        serial = self.tab_main[tab_idx,:]

        input = serial[:self.input_day]
        input = torch.from_numpy(input).to(torch.float32)
        output = serial[self.input_day:self.output_day]
        output = torch.from_numpy(output).to(torch.float32)
        return input, output  # 在这里确定返回，只要调用 类名[i]就返回

    def __len__(self):
        return len(self.list)


if __name__ == '__main__':
    trainset = MyData(root="../../Corvid19Dataset/Dataset64_16_nor.csv", train=True)
    testset = MyData(root="../../Corvid19Dataset/Dataset64_16_nor.csv", train=False)
    # print(trainset[0])
    # print(testset[0])
    # print(len(testset))
    # print(len(trainset))

    input,output = trainset[0]
    print(len(input))
    print(len(output))

    loader = DataLoader(testset, batch_size=4, shuffle=True)  # 注意batch_size 不要超过8，这是验证集，大小只有9
    for data in loader:
        input,output = data
        print(input)
        print(output)

