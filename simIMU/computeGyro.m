function w = computeGyro(q, dt)

    w = angvel(q, dt, 'frame'); % units in rad/s 

end
