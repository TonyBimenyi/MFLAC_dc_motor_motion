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


#DC Motor Parameters

Ra = 0.04  # Armature resistance (Ω)
J = 1.2  # Inertia (kg*m^2)
La = 0.006  # Armature inductance (H)
k = 0.245  # Motor constant (Nm/A^2)
B = 0.00006  # Damping coefficient (kg*m^2/s)
Lf = 1.0  # Field inductance constant
Tl = 25.0  # Load torque (N)
omega_ref = 200.0  # Motor speed reference (rad/s)

def Te(ifield,iarm):
    return k * ifield * iarm


# Loop with time adjustment
for t in range(1000):
     # Use actual time for calculations (t-2)
    #   actual_time = t - 2 
    #Estimator
       
    if(t >= 2):
      
       phi[t] = (phi[t-1]) + ((eta*u[t-1]/mu+u[t-1**2])) * (y[t]-phi[t-1] * u[t-1]) 

       if (phi[t]) <= epsilon or abs(u[t-1]) <= epsilon:
            phi[t] = phi[1]


    #input
    if t >= 2:
        # Check if t+1 is within bounds for ystar
        if t + 1 < 1000:
            u[t] = u[t-1] = ((rho * phi[t]) / ((lamda) + (np.linalg.norm(phi[t]))**2)) * (ystar[t+1] - y[t])

        else:
        # Handle the case when t+1 is out of bounds
            u[t] = ((rho * phi[t]) / lamda + phi[t]**2) * (ystar[t+1] - y[t])

    #DC Motor 

    if(t >= 2):
       




# Plotting with adjusted time labels
# plt.figure(1)
# plt.plot(u, '-g')  # Use time_plot for x-axis
# plt.plot(phi,'--r')
# plt.legend(['input'])
# plt.xlabel('Sampling instant(s/s)')
# plt.ylabel('Controller(V)')
# plt.grid(True)

# Set x-axis limits to clearly show 0 to 2 range
# plt.xlim(0, 2)
# plt.ylim(0, 60)

plt.show()

plt.figure(2)
plt.plot(wref)
plt.xlabel('Sampling instant(s/s)')
plt.ylabel('Rotate speed')
plt.show()
plt.grid(True)
