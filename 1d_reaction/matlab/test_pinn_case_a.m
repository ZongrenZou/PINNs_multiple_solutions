close all
clear all

data = load("pinn_case_a.mat");
% x = data.x_test';
% u = data.u_pred;
% u_x = data.u_x_pred;
x = data.x_test_densest;
u = data.u_pred_densest;
u_x = data.u_x_pred_densest;

%% Check the value of u at x=-0.8 to choose initial guesses for each class
figure(1)
plot(u(:, 21), "k.")
idx1 = 716;
idx2 = 254;

%% Part I: check the residual
options = bvpset("RelTol", eps, "Stats", "on");
solinit1 = bvpinit(x, @guess);
solinit2 = bvpinit(x, @guess);

solinit1.y(1, :) = u(idx1, :);
solinit1.y(2, :) = u_x(idx1, :);
solinit2.y(1, :) = u(idx2, :);
solinit2.y(2, :) = u_x(idx2, :);

sol1 = bvp4c(@bvpfcn, @bcfcn, solinit1, options);
sol2 = bvp4c(@bvpfcn, @bcfcn, solinit2, options);
y1 = sol1.y(1, :);
y2 = sol2.y(1, :);

figure;
hold on
plot(sol1.x, y1, "k-", "LineWidth", 2);
% plot(x, u(idx1, :), "r--", "LineWidth", 2);

plot(sol2.x, y2, "b-", "LineWidth", 2);
% plot(x, u(idx2, :), "r--", "LineWidth", 2);
legend(["u_1", "u_2"])

disp(sol1.stats.maxres)
disp(sol2.stats.maxres)

save case_a_densest x y1 y2 u


function dydx = bvpfcn(x, y) % equation to solve
w = 6;
f_fn = @(x) 0.01 * (6 * sin(w*x) * w * w * cos(w*x) * cos(w*x) - 3 * sin(w*x) * sin(w*x) * w * sin(w*x) * w) + 0.7 * tanh(sin(w*x)^3);
dydx = [y(2)
        100 * (f_fn(x) - 0.7 * tanh(y(1)))];
end
%--------------------------------

function res = bcfcn(ya, yb) % boundary conditions
res = [ya(1) + sin(6)^3
       yb(1) - sin(6)^3];
end
%--------------------------------

function g = guess(x) % initial guess for y and y'
g = [sin(10*x)
     cos(10*x)];
end
