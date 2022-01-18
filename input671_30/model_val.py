
import torch
import numpy as np
import matplotlib.pyplot as plt
from read_data import *

model_path = "nnline_save/linear5/nnline_model10000.pth"
nn_model = torch.load(model_path)
nn_model.to("cpu")

trainset = MyData("Dataset.csv",train=True)
testset = MyData("Dataset.csv",train=False)
print(len(trainset))

us_case = testset[2]
poland_case = trainset[207]
japan_case = trainset[150]
finland_case = trainset[113]

input,target = japan_case
output = nn_model(input)

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
plt.plot(x,y_ture, label="predict", color="r")
plt.plot(x,y_hap, label="ture", color="b")
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


x = range(1,702)
plt.figure(3)
plt.figure(figsize=(10,6), dpi=110)  # 设置画板属性
plt.plot(x,y_input+y_hap, label="predict", color="b")
plt.plot(x,y_input+y_ture, label="ture", color="r")
plt.xlabel("Day")
plt.ylabel("Confirm Case")
plt.title("Corvid19 case")
plt.legend(loc="best")

plt.show()