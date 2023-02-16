import matplotlib.pyplot as plt
import numpy as np
import copy

# X Values
xv = np.linspace(0,10,100)
num_vals = len(xv)

sd = np.zeros((num_vals,1))
Y_hat = np.zeros((num_vals,1))

# Sample Data
for i in range(num_vals):
    sd[i] = 3*xv[i] + 4 + 3*(2*np.random.rand()-1)
    
# Design Matrix
XM = np.zeros((num_vals,2))

# Response Matrix
YM = np.zeros((num_vals,1))

YM = copy.deepcopy(sd)

for i in range(num_vals):
    XM[i,0] = 1
    XM[i,1] = xv[i]

#Normal Equation
b = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(XM),XM)),np.transpose(XM)),YM)

#Predictions
for i in range(num_vals):
    Y_hat[i] = b[0] + b[1]*xv[i]

plt.scatter(xv,sd)
plt.plot(xv,Y_hat,"r-")
plt.show()


