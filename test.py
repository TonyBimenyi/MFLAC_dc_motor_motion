import numpy as np
import matplotlib.pyplot as plt

# Step factor initializations
rho = 0.00007
eta = 1
lamda = 0.1

# input difference
epsilon = 10**(-5)

# Define phi and ystar as arrays
phi = np.zeros((1000, 1))
phi[0] = 0.5
phi[1] = 0.5
mu = 1

# Input initializations
u = np.zeros((1000, 1))
u[0] = 50
u[1] = 50
ystar = np.zeros((1001, 1))
y = np.zeros((1000, 1))

# Define time array with offset (adjusted for plotting)
time_plot = np.arange(0, 1000)  # This creates an array from 0 to 999

# Loop with time adjustment
for t in range(1000):
     # Use actual time for calculations (t-2)
    #   actual_time = t - 2 
    #Estimator
       
    if(t >= 2):
      
       phi[t] = (phi[t-1]) + ((eta*u[t-1]/mu+u[t-1**2])) * (y[t]-phi[t-1] * u[t-1]) 

       if abs(phi[t]) <= epsilon or abs(u[t-1]-u[t-2]) <= epsilon or np.sign(phi[t]) != np.sign(phi[0]):
            phi[t] = phi[0]


    #input
    if t >= 2:
        # Check if t+1 is within bounds for ystar
        if t + 1 < 1000:
            u[t] = u[t-1] = ((rho * phi[t]) / lamda + np.linalg.norm(phi[t])**2) * (ystar[t+1] - y[t])

        else:
        # Handle the case when t+1 is out of bounds
            u[t] = ((rho * phi[t]) / lamda + phi[t]**2) * (ystar[t+1] - y[t])
    




# Plotting with adjusted time labels
plt.plot(u)  # Use time_plot for x-axis
plt.title('Input u over time')
plt.xlabel('Time')
plt.ylabel('Value of u')
plt.grid(True)

# Set x-axis limits to clearly show 0 to 2 range
# plt.xlim(0, 2)
# plt.ylim(0, 60)

plt.show()
