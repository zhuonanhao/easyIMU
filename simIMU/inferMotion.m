% Clear workspace
clear; clc; close all;

% Load the input data from 'input.txt'
% The file should have columns: [time, ax, ay, az, wx, wy, wz]
data = load('input.txt');

% Extract time and input values
time_data = data(:, 1);
ax_data = data(:, 2);
ay_data = data(:, 3);
az_data = data(:, 4);
wx_data = data(:, 5);
wy_data = data(:, 6);
wz_data = data(:, 7);

% Initial state vector q: [x, y, z, psi, theta, phi, vx, vy, vz]
q0 = zeros(9, 1);

% Time span for the simulation
tspan = [min(time_data), max(time_data)];

% Interpolate input values based on the simulation time
ax_func = @(t) interp1(time_data, ax_data, t, 'linear', 'extrap');
ay_func = @(t) interp1(time_data, ay_data, t, 'linear', 'extrap');
az_func = @(t) interp1(time_data, az_data, t, 'linear', 'extrap');
wx_func = @(t) interp1(time_data, wx_data, t, 'linear', 'extrap');
wy_func = @(t) interp1(time_data, wy_data, t, 'linear', 'extrap');
wz_func = @(t) interp1(time_data, wz_data, t, 'linear', 'extrap');

% Define the dynamics function handle with time-varying inputs
dynamics = @(t, q) poseDynamics(q, ax_func(t), ay_func(t), az_func(t), wx_func(t), wy_func(t), wz_func(t));

% Solve the ODE
[t, q] = ode45(dynamics, tspan, q0);

figure;
plot3(q(:, 1), q(:,2), q(:,3)); % Position (x, y, z)

% % Plot the results
% figure;
% subplot(3, 1, 1);
% plot(t, q(:, 1:3)); % Position (x, y, z)
% title('Position');
% xlabel('Time (s)');
% ylabel('Position (m)');
% legend('x', 'y', 'z');

% subplot(3, 1, 2);
% plot(t, q(:, 4:6)); % Orientation (psi, theta, phi)
% title('Orientation');
% xlabel('Time (s)');
% ylabel('Angles (rad)');
% legend('\psi', '\theta', '\phi');

% subplot(3, 1, 3);
% plot(t, q(:, 7:9)); % Velocity (vx, vy, vz)
% title('Velocity');
% xlabel('Time (s)');
% ylabel('Velocity (m/s)');
% legend('vx', 'vy', 'vz');

function dqdt = poseDynamics(q, ax, ay, az, wx, wy, wz)
    % Extract state variables
    x = q(1); 
    y = q(2);
    z = q(3);
    psi = q(4);
    theta = q(5);
    phi = q(6);
    vx = q(7);
    vy = q(8);
    vz = q(9);

    % Gravitational acceleration
    g = [0; 0; 9.81];

    % Define accelerations and angular velocities in body frame
    a_body = [ax; ay; az];
    w_body = [wx; wy; wz];

    % Rotation matrix for orientation dynamics
    A = [0, sin(phi) * sec(theta), cos(phi) * sec(theta);
         0, cos(phi), -sin(phi);
         1, sin(phi) * tan(theta), cos(phi) * tan(theta)];

    % Rotation matrices for transforming accelerations
    M1 = [cos(psi), -sin(psi), 0;
          sin(psi), cos(psi), 0;
          0, 0, 1];
    M2 = [cos(theta), 0, sin(theta);
          0, 1, 0;
          -sin(theta), 0, cos(theta)];
    M3 = [1, 0, 0;
          0, cos(phi), -sin(phi);
          0, sin(phi), cos(phi)];

    % Compute derivatives
    dqdt = [vx; 
            vy; 
            vz; 
            A * w_body; 
            M1 * M2 * M3 * a_body - g];
end