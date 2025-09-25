clear all
close all
% 
% data = load("results3.mat");
% x_test = double(data.xx);
% y_test = double(data.yy);
% u_pred = double(data.uu);


x = linspace(0, 1, 200);
y = linspace(0, 1, 200);
[xx, yy] = meshgrid(x, y);
xxx = reshape(xx, [40000, 1]);
yyy = reshape(yy, [40000, 1]);
location = struct();
location.x = xxx;
location.y = yyy;

for i = 1: 100
    figure
    u_init = pinn(location, i);
    u_init = reshape(u_init, [200, 200]);
    contour(xx, yy, u_init)
    axis("equal")
end



function out = pinn(location, i)

x = location.x;
y = location.y;

%% Load parameters of NNs
data = load("params_case2.mat");
theta = data.theta;
idx = i; % 24, 31, 47, 79
theta = double(theta(idx, :));

st = 1; ed = 100;
w1 = reshape(theta(:, st:ed), [50, 2])';
st = ed + 1; ed = ed + 50*50;
w2 = reshape(theta(:, st:ed), [50, 50])';
st = ed + 1; ed = ed + 50*1;
w3 = reshape(theta(:, st:ed), [50, 1]);
st = ed + 1; ed = ed + 1*50;
b1 = reshape(theta(:, st:ed), [1, 50]);
st = ed + 1; ed = ed + 1*50;
b2 = reshape(theta(:, st:ed), [1, 50]);
st = ed + 1; ed = ed + 1*1;
b3 = reshape(theta(:, st:ed), [1, 1]);

x = reshape(x, [length(x), 1]);
y = reshape(y, [length(y), 1]);

out = [x, y];
out = tanh(out * w1 + b1);
out = tanh(out * w2 + b2);
out = out * w3 + b3;
% out = 1 * out;
% out = out .* (1 - x.^2 - y.^2);
% out = 1 * out + 0.00 * randn(size(out));

end

