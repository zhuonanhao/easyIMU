import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_acc(position, quaternions, dt, v0):
    """
    Compute accelerometer readings (a_imu) from position and quaternion.
    """
    g = np.array([0, 0, -9.81])  # Gravitational acceleration (m/s^2)
    num_steps = len(position)
    acc = np.zeros((num_steps, 3))
    acc[0, :] = -g
    velocity_previous = np.array(v0)
    
    for i in range(1, num_steps):
        # Compute velocity
        velocity = (position[i, :] - position[i - 1, :]) / dt
        
        # Compute acceleration in the world frame
        a_world = (velocity - velocity_previous) / dt
        
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternions[i].rotation_matrix
        
        # Compute accelerometer reading in IMU frame
        a_local = np.dot(rotation_matrix, (a_world - g))
        
        # Store the accelerometer reading
        acc[i, :] = a_local
        
        # Update the previous velocity
        velocity_previous = velocity

    return acc

def compute_gyro(quaternions, dt):
    """
    Compute angular velocity (gyroscope readings) in rad/s.

    Parameters:
        quaternions (list or numpy.ndarray): Sequence of quaternion objects representing orientations.
        dt (float): Time step between consecutive quaternions in seconds.

    Returns:
        numpy.ndarray: Angular velocities (Nx3 array), where each row corresponds to angular velocity (wx, wy, wz) at each time step.
    """
    if len(quaternions) < 2:
        raise ValueError("At least two quaternions are required to compute angular velocities.")
    
    angular_velocities = []
    
    for i in range(1, len(quaternions)):
        # Compute the change in orientation as a quaternion
        delta_q = quaternions[i] * quaternions[i - 1].inverse
        
        # Calculate the angular velocity vector in rad/s
        w = 2 * np.array(delta_q.axis) * delta_q.angle / dt
        angular_velocities.append(w)
    
    # Include zero angular velocity at t=0
    angular_velocities = np.vstack((np.zeros(3), angular_velocities))
    
    return angular_velocities


# Parameters
plot_flag = False
axis_length = 5
dt = 0.01
t_max = 3
t = np.arange(0, t_max + dt, dt)

# Generate non-constant velocity
velocity_x = 2 * np.sin(2 * t)
velocity_y = -3 * np.sin(3 * t)
velocity_z = 1 * np.sin(1 * t)
v0 = np.array([velocity_x[0], velocity_y[0], velocity_z[0]])

# Integrate to get position
position_x = cumulative_trapezoid(velocity_x, t, initial=0)
position_y = cumulative_trapezoid(velocity_y, t, initial=0)
position_z = cumulative_trapezoid(velocity_z, t, initial=0)
position = np.column_stack((position_x, position_y, position_z))

# Generate sinusoidal Euler angles
eulerAngles_x = 45 * np.sin(5 * t)
eulerAngles_y = 90 * np.sin(3 * t)
eulerAngles_z = 30 * np.sin(2 * t)
eulerAngles = np.column_stack((eulerAngles_z, eulerAngles_y, eulerAngles_x))

# Convert Euler angles to quaternions
quaternions = [Quaternion(matrix=R.from_euler('ZYX', angles, degrees=True).as_matrix()) for angles in eulerAngles]

# Compute IMU readings
local_acc = compute_acc(position, quaternions, dt, v0)
local_gyro = compute_gyro(quaternions, dt)
imu_reading = np.hstack((local_acc, local_gyro))

# Pose data
pose = np.hstack((position, eulerAngles))

# Save to files
# np.savetxt('input.txt', np.column_stack((t, imu_reading)), fmt='%.6f', delimiter=',')
# np.savetxt('output.txt', np.column_stack((t, pose)), fmt='%.6f', delimiter=',')

np.savetxt('dataset2.txt', np.column_stack((t, imu_reading, pose)), fmt='%.6f', delimiter=',')

# Plot results
if plot_flag:
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    for i, quat in enumerate(quaternions):
        ax.cla()
        ax.quiver(0, 0, 0, *position[i], length=1, normalize=True)
        ax.set_xlim([-axis_length, axis_length])
        ax.set_ylim([-axis_length, axis_length])
        ax.set_zlim([-axis_length, axis_length])
        ax.set_title(f"t = {t[i]:.2f}")
        plt.pause(0.01)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

ax.plot(position_x, position_y, position_z, linewidth=3)

# Set axis limits
ax.set_xlim([-axis_length, axis_length])
ax.set_ylim([-axis_length, axis_length])
ax.set_zlim([-axis_length, axis_length])

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add title and grid
ax.set_title('Trajectory')
ax.grid()

plt.show()
