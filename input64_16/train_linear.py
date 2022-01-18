import torch.optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
from nn_model import *
from read_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("../../Corvid19log/Input64_16/nnline_train/linear2")
model_save_path = "nnline_save/linear2"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

root = "../../Corvid19Dataset/Dataset64_16_nor.csv"
trainset = MyData(root=root, train=True)
print("Train set read: successful.")
testset = MyData(root=root, train=False)
print("Test set read: successful.")

train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=16, shuffle=True)

nnline = nnLinear()
nnline.to(device)

# 损失函数与优化器
loss_fun = nn.L1Loss()
loss_fun.to(device)

learn_rate = 0.001
nnline_optim = torch.optim.Adam(nnline.parameters(), lr=learn_rate)

train_num = 0
test_num = 0
epoch = 4000

start_time = time.time()   # 记录时间
for i in range(epoch):
    # 开始训练
    nnline.train()
    print("--------EPOCH {}--------".format(i+1))
    loss_train = 0

    for data in train_dataloader:
        input, target = data
        input = input.to(device)
        target = target.to(device)

        output = nnline(input)
        loss = loss_fun(output, target)
        loss_train += loss

        # 优化器
        nnline_optim.zero_grad()
        loss.backward()
        nnline_optim.step()

        train_num += 1
        if train_num % 500 == 0:
            print("In train num {}, loss: {}".format(train_num, loss))
            writer.add_scalar(tag="train_num vs loss", scalar_value=loss, global_step=train_num)
    print("In epoch {}, TrainSet train loss: {}".format(i + 1, loss_train))
    writer.add_scalar(tag="epoch(TrainSet) vs loss", scalar_value=loss_train, global_step=i + 1)

    nnline.eval()
    with torch.no_grad():
        loss_test = 0
        for data in test_dataloader:
            input, target = data
            input = input.to(device)
            target = target.to(device)

            output = nnline(input)
            loss_test += loss_fun(output, target)

            test_num += 1

        print("In epoch {}, TestSet test loss: {}".format(i + 1, loss_test))
        writer.add_scalar(tag="epoch(TestSet) vs loss", scalar_value=loss_test, global_step=i + 1)

    if i != 0 and (i+1) % 300 == 0:
        path = os.path.join( model_save_path,("nnline_model{}.pth".format(i + 1)) )
        torch.save(nnline, path )  # 自动保存
    end_time = time.time()
    print("--------Time consume: {}".format(end_time - start_time))

writer.close()
