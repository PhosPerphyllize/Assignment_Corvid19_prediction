
from nn_model import *
from read_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "../../Corvid19Dataset/Dataset64_16_nor.csv"
trainset = MyData(root=root, train=True)
print("Train set read: successful.")
testset = MyData(root=root, train=False)
print("Test set read: successful.")

train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=16, shuffle=True)

nnline1 = torch.load("nnline_save/linear1/nnline_model3900.pth")
nnline1.to(device)
nnline2 = torch.load("nnline_save/linear2/nnline_model3900.pth")
nnline2.to(device)
nnline3 = torch.load("nnline_save/linear3/nnline_model4000.pth")
nnline3.to(device)
nnline4 = torch.load("nnline_save/linear4/nnline_model3200.pth")
nnline4.to(device)
nnline5 = torch.load("nnline_save/linear5_drop/nnline_model4000.pth")
nnline5.to(device)
nnline6 = torch.load("nnline_save/linear6_drop/nnline_model4000.pth")
nnline6.to(device)

loss_fun = nn.L1Loss()
loss_fun.to(device)

nnline5.train()
nnline6.train()
loss_train = [0,0,0,0,0,0]
loss_test = [0,0,0,0,0,0]
output = [0,0,0,0,0,0]
for data in train_dataloader:
    input, target = data
    input = input.to(device)
    target = target.to(device)

    output[0] = nnline1(input)
    output[1] = nnline2(input)
    output[2] = nnline3(input)
    output[3] = nnline4(input)
    output[4] = nnline5(input)
    output[5] = nnline6(input)
    for i in range(6):
        loss = loss_fun(output[i], target)
        loss_train[i] += loss

nnline5.eval()
nnline6.eval()
with torch.no_grad():
    for data in test_dataloader:
        input, target = data
        input = input.to(device)
        target = target.to(device)

        output[0] = nnline1(input)
        output[1] = nnline2(input)
        output[2] = nnline3(input)
        output[3] = nnline4(input)
        output[4] = nnline5(input)
        output[5] = nnline6(input)
        for i in range(6):
            loss = loss_fun(output[i], target)
            loss_test[i] += loss



for i in range(len(loss_train)):
    print("In linear {}, TrainSet loss: {}".format(i+1, loss_train[i]))
    print("In linear {}, TestSet loss: {}".format(i+1, loss_test[i]))

