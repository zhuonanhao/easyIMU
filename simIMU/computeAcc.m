function acc = computeAcc(position, q, dt, v0)
    % Function to compute accelerometer readings (a_imu) from position and quaternion
    
    % Constants
    g = [0, 0, -9.81]; % Gravitational acceleration (m/s^2)
    
    % Number of steps
    num_steps = length(position);
    
    % Pre-allocate accelerometer readings (IMU frame)
    acc = zeros(num_steps, 3);
    acc(1, :) = -g;
    
    % Initialize previous velocity
    velocity_previous = v0;
    
    % Loop through each time step
    for i = 2:num_steps
        % Compute velocity (difference between consecutive positions)
        velocity = (position(i,:) - position(i-1,:)) / dt;

        % Compute acceleration in the world frame (difference in velocity)
        a_world = (velocity - velocity_previous) / dt;

        % Convert quaternion to rotation matrix
        R = quat2rotm(q(i,:).');
        
        % Compute accelerometer reading in IMU frame (without gravity)
        a_local = R * (a_world' - g'); % Accelerometer reading in IMU frame
        
        % Store the accelerometer reading
        acc(i, :) = a_local';
        
        % Update the previous velocity for the next iteration
        velocity_previous = velocity;
    end
    
end
