import numpy as np
import matplotlib.pyplot as plt

# Step factor initializations
rho = 0.01
eta = 0.01
lamda = 1

# input difference
epsilon = 10**(-5)

# Define phi and ystar as arrays
phi = np.zeros((1000, 1))
phi[0] = 1
phi[1] = 1
mu = 1

# Input initializations
u = np.zeros((1000, 1))
u[0] = 1
u[1] = 1


yt = np.zeros((1001, 1))
y = np.zeros((1001, 1))
e = np.zeros((1001, 1))
y[0] = 0
y[1] = 0
# Define time array with offset (adjusted for plotting)
time_plot = np.arange(0, 1000)  # This creates an array from 0 to 999


#DC Motor Parameters

Ra = 0.04  # Armature resistance (Ω)
J = 1.2  # Inertia (kg*m^2)
La = 0.006  # Armature inductance (H)
k_ = 0.245  # Motor constant (Nm/A^2)
B = 0.00006  # Damping coefficient (kg*m^2/s)
Lf = 1.0  # Field inductance constant
Tl = 25.0  # Load torque (N)
omega_ref = 200.0  # Motor speed reference (rad/s)
# ua = 20
delta  = 3

def model(u,ddtheta,i):
    ddtheta = (-B/J)*ddtheta+(k_/J)*i
    di = (-k_/La)*ddtheta-(Ra/La)*i + u/La
    return di 


for k in range(1001):
     # Trajectory
    # if k <= 500:
    #     yt[k+1] = 125 * (-1)**(round(k/100))
    # elif 500 < k <= 700:
    #     yt[k+1] = 125 * np.sin(k*np.pi/100) + 0.3 * np.cos(k*np.pi/50)
    # else:
    #     yt[k+1] = 125* (-1)**(round(k/500))
    yt[k]=30*np.sin(0.02*k)+140


# Loop with time adjustment
for k in range(1000):
     # Use actual time for calculations (t-2)
    #   actual_time = t - 2 

    
    
    #Estimator
       
    if k==0 :
        phi[0]=1
    elif k == 1:
       phi[k] = phi[k-1] + ((eta*(u[k-1]-0)) / (mu + (u[k-1]-0)**2)) * (y[k]-y[k-1] - phi[k-1]*(u[k-1]-0))
    else:
       phi[k] = phi[k-1] + ((eta*(u[k-1]-u[k-2])) / (mu + (u[k-1]-u[k-2])**2)) * (y[k]-y[k-1] - phi[k-1]*(u[k-1]-u[k-2]))

    # if (phi[k]) <= epsilon or abs(u[k-1]-u[k-2]) <= epsilon:
    #     phi[k] = phi[1]


    #input 
    if k == 0:
        u[0]=1
    else:
        # Check if k+1 is within bounds for yt
        # if k + 1 < 1000:
        u[k] = u[k-1] + (rho * phi[k]) / (lamda + ((np.linalg.norm(phi[k]))**2)) * (yt[k+1] - y[k])

        # else:
        # # Handle the case when t+1 is out of bounds
        #     u[k] = ((rho * phi[k]) / lamda + phi[k]**2) * (yt[k+1] - y[k])



    #if 1 <= k <= 499 :
    # if k >= 2:
    #y[k+1] = (y[k] /  (1 + y[k]**2)) + 5*u[k]
    y[k+1] = model(u[k],5,2)
    e[k+1] = yt[k+1] - y[k+1]





#    for u, ddtheta, i in range (1000) :

#        ddtheta_values = model(u,ddtheta, i)
    


    #  System Dynamics
    # if 1 <= t <= 499:  
    #     y[k+1] = (y[k]/(1+(y[k])**2))+(u[k]**3)
    # elif k >= 500:  
    #     y[k+1] = (y[k]*y[k-1]*y[k-2]*u[k-1]*(y[k-2]-1)+round(t/500)*u[k])/(1+(y[k-1]**2)+(y[k-2]**2))
    #DC Motor 

    # def motor_speed(t):
    #     if k >= 2:
    #         tf = k / ((J + B) * (La + Ra) + k**2)
    #         s = tf
    #         return (k * omega_ref) / ((J*s + Ra) * (La*s + Ra) + k**2)
    #     else:
    #             return 0  # Return 0 for k < 2
        


        # def Te(ifield,iarm):
        #     return K * ifield * iarm

        # def E(ifield):
        #  return k * ifield * omega_ref

        # K = k * Lf

        # def dia_dt(ua,ia):
        #     (1/La) * (ua - Ra * ia - E)

        # def dif_dt(uf,Rf, ifield):
        #     (1/Lf) * (uf - Rf * ifield)

        # domega_dt = (1/J) * ((Te) - (Tl))
        # print(motor_speed)
    # Define time array
# t_values = np.linspace(0, 10, 1000)  # Adjust time range as needed

# # Calculate motor speed for each time value
# motor_speed_values = [motor_speed(t) for k in t_values]



# Plotting with adjusted time labels
plt.figure(1)

# plt.plot(phi,'--r')
# plt.plot(yt,'--b')
plt.plot(y,'--k')
plt.plot(yt,'--b')
plt.legend(['input'])
plt.xlabel('Sampling instant(s/s)')
plt.ylabel('Controller(V)')
plt.grid(True)

# Set x-axis limits to clearly show 0 to 2 range
# plt.xlim(0, 2)
# plt.ylim(0, 60)

plt.show()

plt.figure(2)
plt.plot(u,'-g')  # Use time_plot for x-axis
# plt.plot(t_values,motor_speed_values, '--g')
# plt.xlabel('Sampling instant(s/s)')
# plt.ylabel('Rotate speed')

# plt.grid(True)
plt.show()
plt.figure(3)
plt.plot(e,'-g') 
plt.show()

plt.figure(4)
plt.plot(phi,'-r') 
plt.show()