function p = load_mmaud(drone, dt)
d = readmatrix(fullfile(data_folder(), drone + ".csv"), "NumHeaderLines", 1);
tt = (0:ceil(d(end, 1) / dt) - 1)' * dt;
p = interp1(d(:, 1), d(:, 2:4), tt);
end
