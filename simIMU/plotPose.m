function plotPose(position, quaternion)

clf

% Convert the quaternion to a rotation matrix
R = quat2rotm(quaternion);

% Define the unit vectors along the X, Y, Z axes
unit_vectors = [1 0 0; 0 1 0; 0 0 1]; % X, Y, Z axes as row vectors

% Apply the rotation to each unit vector
rotated_vectors = (R * unit_vectors')';

% Create a figure and plot the original and rotated vectors
hold on;

% Plot the original vectors (before rotation), starting from the position
quiver3(position(1), position(2), position(3), unit_vectors(1, 1), unit_vectors(1, 2), unit_vectors(1, 3), 'r', 'LineWidth', 2); % X axis (original)
quiver3(position(1), position(2), position(3), unit_vectors(2, 1), unit_vectors(2, 2), unit_vectors(2, 3), 'g', 'LineWidth', 2); % Y axis (original)
quiver3(position(1), position(2), position(3), unit_vectors(3, 1), unit_vectors(3, 2), unit_vectors(3, 3), 'b', 'LineWidth', 2); % Z axis (original)

% Plot the rotated vectors, starting from the position
quiver3(position(1), position(2), position(3), rotated_vectors(1, 1), rotated_vectors(1, 2), rotated_vectors(1, 3), 'r--', 'LineWidth', 2); % X axis (rotated)
quiver3(position(1), position(2), position(3), rotated_vectors(2, 1), rotated_vectors(2, 2), rotated_vectors(2, 3), 'g--', 'LineWidth', 2); % Y axis (rotated)
quiver3(position(1), position(2), position(3), rotated_vectors(3, 1), rotated_vectors(3, 2), rotated_vectors(3, 3), 'b--', 'LineWidth', 2); % Z axis (rotated)

% Highlight the origin (0,0,0)
plot3(0, 0, 0, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8); % Black point at the origin

hold off;

end