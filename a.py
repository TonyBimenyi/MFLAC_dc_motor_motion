import numpy as np
import matplotlib.pyplot as plt

# Constants
Ra = 0.04  # Armature resistance (Ω)
J = 1.2  # Inertia (kg*m^2)
La = 0.006  # Armature inductance (H)
k = 0.245  # Motor constant (Nm/A^2)
B = 0.00006  # Damping coefficient (kg*m^2/s)
Lf = 1.0  # Field inductance constant
TL = 25.0  # Load torque (N)
omega_ref = 200.0  # Motor speed reference (rad/s)

# Time parameters
t_start = 0
t_end = 10
dt = 0.01
time = np.arange(t_start, t_end, dt)

# Define the equations
def Te(ifield, iarm):
    return k * ifield * iarm

def E(ifield, omega):
    return k * ifield * omega

def difdt(uf, ifield):
    return (1 / Lf) * (uf - Ra * ifield)

def dwdt(Te, TL):
    return (1 / J) * (Te - TL)

# Initial conditions
ifield_0 = 1.0  # Initial field current
omega_0 = 0.0  # Initial motor speed
Te_0 = Te(ifield_0, ifield_0)
E_0 = E(ifield_0, omega_0)

# Initialize arrays to store results
ifield_values = np.zeros_like(time)
omega_values = np.zeros_like(time)
Te_values = np.zeros_like(time)
E_values = np.zeros_like(time)

# Simulate the system
ifield_values[0] = ifield_0
omega_values[0] = omega_0
Te_values[0] = Te_0
E_values[0] = E_0

for i in range(1, len(time)):
    Te_values[i] = Te(ifield_values[i-1], ifield_values[i-1])
    E_values[i] = E(ifield_values[i-1], omega_values[i-1])
    difdt_val = difdt(Te_values[i-1], ifield_values[i-1])
    dwdt_val = dwdt(Te_values[i-1], TL)
    ifield_values[i] = ifield_values[i-1] + difdt_val * dt
    omega_values[i] = omega_values[i-1] + dwdt_val * dt

# Plot the results
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(time, ifield_values)
plt.title('Field Current vs Time')
plt.xlabel('Time')
plt.ylabel('Field Current')

plt.subplot(2, 2, 2)
plt.plot(time, omega_values)
plt.title('Motor Speed vs Time')
plt.xlabel('Time')
plt.ylabel('Motor Speed')

plt.subplot(2, 2, 3)
plt.plot(time, Te_values)
plt.title('Developed Torque vs Time')
plt.xlabel('Time')
plt.ylabel('Developed Torque')

plt.subplot(2, 2, 4)
plt.plot(time, E_values)
plt.title('Back EMF vs Time')
plt.xlabel('Time')
plt.ylabel('Back EMF')

plt.tight_layout()
plt.show()
