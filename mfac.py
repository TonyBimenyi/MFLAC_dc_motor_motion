import numpy as np
import matplotlib.pyplot as plt

# step factors initialization
rho = 0.00007
eta = 0.5

# input difference
epsilon = 10**(-5)

# weighing factors initialization
lambda_ = 0.1
mu = 1

# input initialization
u = np.zeros((500, 1))

# parameter initialization
phi = np.zeros((500, 1))
phi[0] = 0.5
phi[1] = 0.5

# System Dynamics initialization
y = np.zeros((1001, 1))  # Increase the size of y array by 1
y[0] = -1
y[1] = 1

# Trajectory initilization
yt = np.zeros((1001, 1))

for k in range(500):
    
    # Trajectory
    if k <= 300:
        yt[k+1] = 0.5 * (-1)**(round(k/500))
    elif 300 < k <= 700:
        yt[k+1] = 0.5 * np.sin(k*np.pi/100) + 0.3 * np.cos(k*np.pi/50)
    else:
        yt[k+1] = 0.5 * (-1)**(round(k/500))
      
    # Parameter Estimate
    if k >= 2:
        phi[k] = phi[k-1] + ((eta*(u[k-1]-u[k-2]))/(mu + (u[k-1]-u[k-2])**2))* (y[k]-y[k-1] - phi[k-1]*(u[k-1]-u[k-2]))
        
        if abs(phi[k]) <= epsilon or abs(u[k-1]-u[k-2]) <= epsilon or np.sign(phi[k]) != np.sign(phi[0]):
            phi[k] = phi[0]
   
    # Input Calculation
    if k >= 2:
        u[k] = u[k-1] + (rho*phi[k]/(lambda_+np.linalg.norm(phi[k])**2))*(yt[k+1]-y[k])
    
    # System Dynamics
    if 1 <= k <= 499:  
        y[k+1] = (y[k]/(1+(y[k])**2))+(u[k]**3)
    elif k >= 500:  
        y[k+1] = (y[k]*y[k-1]*y[k-2]*u[k-1]*(y[k-2]-1)+round(k/500)*u[k])/(1+(y[k-1]**2)+(y[k-2]**2))



plt.plot(yt[1:], '--r')
plt.plot(y)
plt.title('SISO CFDL MFAC Trajectory Tracking')
plt.legend(['trajectory', 'tracking'])
plt.show()
