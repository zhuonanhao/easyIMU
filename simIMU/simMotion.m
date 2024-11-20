clear;clc;close all

plotFlag = 1;
axis_length = 5;

% Time
dt = 0.01;
t_max = 3;
t = 0:dt:t_max;

% Generate non-constant velocity (sinusoidal example)
velocity_x = 2 * sin(2 * t); % Sinusoidal variation for X velocity
velocity_y = -3 * sin(3 * t); % Sinusoidal variation for Y velocity
velocity_z = 1 * sin(1 * t);   % Sinusoidal variation for Z velocity
v0 = [velocity_x(1), velocity_y(1), velocity_z(1)];

% Integrate to get the position (starting at position [0, 0, 0])
position_x = cumtrapz(t, velocity_x); % Cumulative sum to get positionv0
position_y = cumtrapz(t, velocity_y);
position_z = cumtrapz(t, velocity_z);

% Combine the position vectors
position = [position_x', position_y', position_z'];

% Generate sinusoidal pattern for Euler angles
eulerAngles_x = 45 * sin(5 * t);  % Oscillating between -45 and 45 degrees
eulerAngles_y = 90 * sin(3 * t);  % Oscillating between -90 and 90 degrees
eulerAngles_z = 30 * sin(2 * t);  % Oscillating between -30 and 30 degrees

% Combine into a matrix for eulerAngles
eulerAngles = [eulerAngles_x', eulerAngles_y', eulerAngles_z'];
q = quaternion(eulerAngles,'eulerd','ZYX','frame');

local_w = computeGyro(q, dt);
local_acc = computeAcc(position, q, dt, v0);
imu_reading = [local_acc local_w];
pose = [position eulerAngles];

% Save time and state trajectory data to a text file
filename = 'input.txt';

% Write to a text file
writematrix([t' imu_reading], filename);

% Save time and state trajectory data to a text file
filename = 'output.txt';

% Write to a text file
writematrix([t' pose], filename);

if plotFlag == 1
    figure(1)
    
    for i = 1:length(q)
        
        plotPose(position(i,:), q(i,:))
        view(45,45)
    
        axis equal;
        xlim([-axis_length axis_length])
        ylim([-axis_length axis_length])
        zlim([-axis_length axis_length])
    
        xlabel('X')
        ylabel('Y')
        zlabel('Z')
        grid on
        box on
        title(sprintf('t = %.2f', t(i)))
        drawnow
    end
end


figure(2)
plot3(position_x, position_y, position_z, 'LineWidth',3)

axis equal;
xlim([-axis_length axis_length])
ylim([-axis_length axis_length])
zlim([-axis_length axis_length])
xlabel('X')
ylabel('Y')
zlabel('Z')
grid on
box on
title('Trajectory')