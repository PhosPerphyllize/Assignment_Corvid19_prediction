
import numpy as np

tab_main = np.genfromtxt("../../Corvid19Dataset/Dataset.csv", delimiter=",")

days_input = 64
days_output = 16
days_total = days_output + days_input
day_sum = 701
tab_output = np.zeros([1, days_total])
regions = len(tab_main[:,0])

for j in range(regions):
    line_cho = tab_main[j,:]
    tab_temp = np.zeros([1, days_total])
    for i in range(day_sum - days_total + 1):
        line_stack = line_cho[i:i + days_total]
        tab_temp = np.vstack((tab_temp,line_stack))
    tab_output = np.vstack((tab_output,tab_temp))
    print("Solve line {} over.".format(j))
    # if j!=0 and j%50== 0:
    #     np.savetxt("Dataset{}_{}.csv".format(days_input, j), tab_output, delimiter=',')

np.savetxt("../../Corvid19Dataset/Dataset{}_{}.csv".format(days_input, days_output), tab_output, delimiter=',')