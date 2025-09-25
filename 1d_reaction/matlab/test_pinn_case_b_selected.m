close all
clear all

data = load("pinn_case_b_selected.mat");
x = data.x_test_densest;
u = data.u_pred_densest;
u_x = data.u_x_pred_densest;

figure;
hold on
for i = 1: 9
    plot(x, u(i, :), "k-", "LineWidth", 1);
end


%% Part I: check the residual
options = bvpset("RelTol", eps, "Stats", "on");
solinit = bvpinit(x, @guess);
ys = zeros(9, 6401);
figure;
hold on
for i = 1: 9
    solinit.y(1, :) = u(i, :);
    solinit.y(2, :) = u_x(i, :);
    sol = bvp4c(@bvpfcn, @bcfcn, solinit, options);
    disp([i])
    disp(max(abs(sol.x - x')))
    disp([sol.stats.maxres])
    plot(sol.x, sol.y(1, :), "LineWidth", 1);
    ys(i, :) = sol.y(1, :);
end
legend(["$u_1$", "$u_2$", "$u_3$", "$u_4$", "$u_5$", "$u_6$", "$u_7$", "$u_8$", "$u_9$"])

save case_b_densest x ys


function dydx = bvpfcn(x, y) % equation to solve
w = 10;
f_fn = @(x) 0.01 * (6 * sin(w*x) * w * w * cos(w*x) * cos(w*x) - 3 * sin(w*x) * sin(w*x) * w * sin(w*x) * w) + 0.7 * tanh(sin(w*x)^3);
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
