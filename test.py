import numpy as np
import matplotlib.pyplot as plt

#step_factor_initializations

rho = 0.00007
eta =0.5
lamda = 2
phi = 0.5 

#input initializations

u = np.zeros((1000,1))
ystar = np.zeros((1000,1))

for t in range(1000):

    #Controller MDAC
    if t >= 500:
        u[t] = u[t-1] = ((rho*phi[t])/lamda+phi[t]**2)*(ystar[t+1]-y[t])



# Plotting
plt.plot(rho)
plt.title('Value of phi1')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()