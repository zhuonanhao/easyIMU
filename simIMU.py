import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Helix parameters
R = 1.0       # Radius of the helix
omega = 2.0   # Angular frequency
c = 0.2       # Pitch
T = 10        # Total time
dt = 0.01     # Time step

# Time vector
t = np.arange(0, T, dt)

# Define position along helix
x = R * np.cos(omega * t)
y = R * np.sin(omega * t)
z = c * t

# Compute velocities
vx = -R * omega * np.sin(omega * t)
vy = R * omega * np.cos(omega * t)
vz = np.full_like(t, c)

# Compute accelerations
ax = -R * omega**2 * np.cos(omega * t)
ay = -R * omega**2 * np.sin(omega * t)
az = np.zeros_like(t)

# Stack IMU data: [ax, ay, az, wx, wy, wz]
imu_data = np.vstack([ax, ay, az, np.gradient(vx)/dt, np.gradient(vy)/dt, np.gradient(vz)/dt]).T

# Plot the 3D helical path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label="Helical Path")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
plt.title("3D Helical Trajectory")
plt.legend()
plt.show()

# Print sample IMU readings
print("Sample IMU Data (Acceleration and Angular Velocity):\n", imu_data[:5])  # Print first 5 samples
