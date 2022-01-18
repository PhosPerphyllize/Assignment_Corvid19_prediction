
import torch
import numpy as np
import matplotlib.pyplot as plt
from read_data import *

model_path = "nnline_save/linear5_drop/nnline_model4000.pth"
# model_path = "nnconv_save/conv2_linear2/nnconv_model3000.pth"
nn_model = torch.load(model_path)
print(nn_model)

nn_model.to("cpu")

trainset = MyData("ValSet64_16.csv",train=True)

finland_case = trainset[0]
hubei_case = trainset[1]
japan_case = trainset[2]
korea_case = trainset[3]
greece_case = trainset[4]
germany_case = trainset[5]
poland_case = trainset[6]
russia_case = trainset[7]
us_case = trainset[8]

input,target = hubei_case

a = input[0]
b = input[-1]

input = input - a
input = input/(b-a)
input = input.view(1,-1)   # 转格式

print(type(input))
print(input)

nn_model.eval()
output = nn_model(input)

print(type(output))
print(output)

input = input.view(-1)
output = output.view(-1)
input = input * (b-a)
input = input + a
output = output * (b-a)
output = output + a

print(type(input))
print(input)
print(type(output))
print(output)

# x = np.arange(0,701)
# y_input = np.array(input)
# y_ture = np.array(target)
# y_hap = np.array(output)

y_input = []
y_ture = []
y_hap = []
y_error = []
for i in range(len(input)):
    y_input.append(input[i].item())
for i in range(len(target)):
    y_ture.append(target[i].item())
for i in range(len(output)):
    y_hap.append(output[i].item())
    y_error.append((y_hap[i]-y_ture[i])/y_ture[i])

x = range(1,len(target)+1)
plt.figure(1)
plt.figure(figsize=(10,6), dpi=100)  # 设置画板属性
plt.plot(x,y_hap, label="predict", color="b")
plt.plot(x,y_ture, label="ture", color="r")
plt.xlabel("Day")
plt.ylabel("Confirm Case")
plt.title("Corvid19 case")
plt.legend(loc="best")

plt.figure(2)
y_error = np.array(y_error)
plt.figure(figsize=(10,6), dpi=100)  # 设置画板属性
plt.fill_between(x[1:], y_error[1:], 0, where=y_error[1:] >= 0, facecolor='green', interpolate=True, alpha=0.7)
plt.fill_between(x[1:], y_error[1:], 0, where=y_error[1:] <= 0, facecolor='red', interpolate=True, alpha=0.7)
plt.xlabel("Day")
plt.ylabel("Error/Confirm Case")
plt.title("Corvid19 case")


x = range(1,81)
plt.figure(3)
plt.figure(figsize=(10,6), dpi=110)  # 设置画板属性
plt.plot(x,y_input+y_hap, label="predict", color="b")
plt.plot(x,y_input+y_ture, label="ture", color="r")
plt.xlabel("Day")
plt.ylabel("Confirm Case")
plt.title("Corvid19 case")
plt.legend(loc="best")

plt.show()