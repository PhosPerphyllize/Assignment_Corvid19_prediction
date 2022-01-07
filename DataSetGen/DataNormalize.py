import numpy as np

tab_main = np.genfromtxt("../../Corvid19Dataset/Dataset64_16.csv", delimiter=",")
print("file read.")

tab_height = len(tab_main[:,0])
tab_length = len(tab_main[0,:])
tab_output = np.ones([tab_height,tab_length])
tab_main = tab_main + 1

for i in range(tab_height):
    tab_main[i,:] = tab_main[i,:] - tab_main[i,:][0]
    gap = tab_main[i,:][63] - tab_main[i,:][0]
    if gap == 0:
        tab_main[i,:] = tab_main[i,:]/1
    else:
        tab_main[i,:] = tab_main[i,:]/gap
    if i%5000 == 0:
        print("line {} finish.".format(i))
np.savetxt("../../Corvid19Dataset/Dataset64_16_nor.csv", tab_main, delimiter=',')
