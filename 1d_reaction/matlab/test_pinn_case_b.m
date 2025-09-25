close all
clear all

data = load("pinn_case_b.mat");
% x = data.x_test';
% u = data.u_pred;
% u_x = data.u_x_pred;
x = data.x_test_densest;
u = data.u_pred_densest;
u_x = data.u_x_pred_densest;

%% Check the value of u at x=-0.8 to choose initial guesses for each class
% figure(1)
% plot(u(:, 21), "k.")
idx1 = 408;
idx2 = 778;
idx3 = 119;

%% Part I: check the residual
eps_densest = eps;
eps_denser = 1e-4;
options = bvpset("RelTol", eps_densest, "Stats", "on");
solinit1 = bvpinit(x, @guess);
solinit2 = bvpinit(x, @guess);
solinit3 = bvpinit(x, @guess);

solinit1.y(1, :) = u(idx1, :);
solinit1.y(2, :) = u_x(idx1, :);
solinit2.y(1, :) = u(idx2, :);
solinit2.y(2, :) = u_x(idx2, :);
solinit3.y(1, :) = u(idx3, :);
solinit3.y(2, :) = u_x(idx3, :);

sol1 = bvp4c(@bvpfcn, @bcfcn, solinit1, options);
sol2 = bvp4c(@bvpfcn, @bcfcn, solinit2, options);
sol3 = bvp4c(@bvpfcn, @bcfcn, solinit3, options);
y1 = sol1.y(1, :);
y2 = sol2.y(1, :);
y3 = sol3.y(1, :);

figure;
hold on
plot(sol1.x, y1, "k-", "LineWidth", 2);
% plot(x, u(idx1, :), "r--", "LineWidth", 2);

plot(sol2.x, y2, "r-", "LineWidth", 2);
% plot(x, u(idx2, :), "r--", "LineWidth", 2);

plot(sol3.x, y3, "b-", "LineWidth", 2);
% plot(x, u(idx3, :), "r--", "LineWidth", 2);

legend(["u_1", "u_2", "u_3"])

disp(sol1.stats.maxres)
disp(sol2.stats.maxres)
disp(sol3.stats.maxres)

% disp(max(abs(sol1.x(2:end) - sol1.x(1:end-1))))
% disp(max(abs(sol2.x(2:end) - sol2.x(1:end-1))))
% disp(max(abs(sol3.x(2:end) - sol3.x(1:end-1))))
% disp(1/800)

save case_b_densest x y1 y2 y3 u


function dydx = bvpfcn(x, y) % equation to solve
w = 10;
f_fn = @(x) 0.01 * (6*w^2*cos(w*x)^2*sin(w*x) - 3*w^2*sin(w*x)^3) + 0.7 * tanh(sin(w*x)^3);
dydx = [y(2)
        100 * (f_fn(x) - 0.7 * tanh(y(1)))];
end
%--------------------------------

function res = bcfcn(ya, yb) % boundary conditions
w = 10;
res = [ya(1) - sin(-w)^3
       yb(1) - sin(w)^3];
end
%--------------------------------

function g = guess(x) % initial guess for y and y'
g = [sin(10*x)
     cos(10*x)];
end
