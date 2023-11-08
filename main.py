# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate

# def integrate_func(x, y):
#    sm = 0
#    for i in range(1, len(x)):
#        h = x[i] - x[i-1]
#        sm += h * (y[i-1] + y[i]) / 2
#
#    return sm

#-----------
#-Read file-
#-----------
address = "E:\postdoctor-HKUST\Resaerch\Voltage control\VCMA\Experiment\PMA for Hall bar-pad voltage\PMNT_(110)\Sample 3\PMNT_Sample3\PMNT_Sample3"
file = address + "/PMNT(110)-Sample 3-Rxy-0.6A+100uA- (20V)-304 points.csv" #change file name at here
df = pd.read_csv(file)
Hz = df['keithley 2440 cur']*18.5
Rxy = df['keithley 2182a read']/df['keithley 6221 cur']


#-----------
#-histogram-
#-----------
counts, bins, _ = plt.hist(Rxy, bins=10000)
sorted_indices = np.argsort(-counts)
highest_indices = sorted_indices[:20]
highest_bins = bins[highest_indices]

print(highest_indices)

pos = highest_bins[highest_bins > 0]
neg = highest_bins[highest_bins < 0]

maximum = np.mean(pos)
minimum = np.mean(neg)
bias = (minimum + maximum)/2


#------
#-Area-
#------
Nor_Rxy = (Rxy-bias)/(maximum-bias)

Hz_pos = Hz[Hz > 0]
Hz_neg = Hz[Hz < 0]
Rxy_pos = Nor_Rxy[Hz > 0]
Rxy_neg = Nor_Rxy[Hz < 0]

#area = integrate(Hz_neg.tolist(),Rxy_neg.tolist())+integrate(Hz_pos.tolist(),Rxy_pos.tolist())
#print(np.abs(area))

neg_area = integrate.simpson(Rxy_neg.tolist(),Hz_neg.tolist())
pos_area = integrate.simpson(Rxy_pos.tolist(),Hz_pos.tolist())

area = np.abs(neg_area + pos_area)
print("area=", area)

#------
#-Plot-
#------
plt.figure()

plt.subplot(121)
plt.plot(Hz, Rxy)
plt.grid(True)
plt.plot(Hz, bias*np.ones_like(Hz), label=f'bias={bias}')
plt.legend()
plt.xlabel('Hz(mT)')
plt.ylabel('Rxy(Ω)')
plt.title('Data')

plt.subplot(122)
plt.plot(Hz, Nor_Rxy)
plt.text(1.5,.5,f'area={area}',
          bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
          ha='center', va='center')
plt.grid(True)
plt.xlabel('Hz(mT)')
plt.ylabel('Rxy(Ω)')
plt.title('Normalized data')

plt.show()


