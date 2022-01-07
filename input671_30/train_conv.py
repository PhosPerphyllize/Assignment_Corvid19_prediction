
import os
from torch.utils.tensorboard import SummaryWriter
import time
from nn_model import *
from read_data import *

writer = SummaryWriter("../../Corvid19log/Input671_30/nnconv_train/conv2_linear2")
model_save_path = "nnconv_save/conv2_linear2"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = MyData(root="Dataset.csv", train=True)
testset = MyData(root="Dataset.csv", train=False)

train_dataloader = DataLoader(trainset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=1, shuffle=True)

nnconv = nnConv()
nnconv.to(device)

# 损失函数与优化器
loss_fun = nn.L1Loss()
loss_fun.to(device)

learn_rate = 0.001
nnline_optim = torch.optim.Adam(nnconv.parameters(), lr=learn_rate)

train_num = 0
test_num = 0
epoch = 5000

start_time = time.time()   # 记录时间
for i in range(epoch):
    # 开始训练
    nnconv.train()
    print("--------EPOCH {}--------".format(i+1))
    loss_train = 0

    for data in train_dataloader:
        input, target = data
        input = input.to(device)
        target = target.to(device)

        output = nnconv(input)
        loss = loss_fun(output, target)
        loss_train += loss

        # 优化器
        nnline_optim.zero_grad()
        loss.backward()
        nnline_optim.step()

        train_num += 1
        if train_num % 50 == 0:
            print("In train num {}, loss: {}".format(train_num, loss))
        writer.add_scalar(tag="train_num vs loss", scalar_value=loss, global_step=train_num)
    print("In epoch {}, TrainSet train loss: {}".format(i + 1, loss_train))
    writer.add_scalar(tag="epoch(TrainSet) vs loss", scalar_value=loss_train, global_step=i + 1)

    nnconv.eval()
    with torch.no_grad():
        loss_test = 0
        for data in test_dataloader:
            input, target = data
            input = input.to(device)
            target = target.to(device)

            output = nnconv(input)
            loss_test += loss_fun(output, target)

            test_num += 1

        print("In epoch {}, TestSet test loss: {}".format(i + 1, loss_test))
        writer.add_scalar(tag="epoch(TestSet) vs loss", scalar_value=loss_test, global_step=i + 1)

    if i != 0 and (i+1) % 1000 == 0:
        path = os.path.join(model_save_path, ("nnconv_model{}.pth".format(i + 1)))
        torch.save(nnconv, path)  # 自动保存
        end_time = time.time()
        print("Time consume: {}".format(end_time - start_time))

writer.close()
