close all
clear all



%% Training
% data = load("pinn_case_b_1.mat");
% x = data.x_test';
% u = data.u_pred;
% u_x = data.u_x_pred;
% 
% figure;
% hold on
% for i = 1: 1000
%     plot(x, u(i, :))
% end
% 
% options = bvpset("RelTol", 1e-3, "Stats", "off");
% solinit = bvpinit(x, @guess);
% 
% figure;
% hold on
% for i = 1: 1000
%     solinit.y(1, :) = u(i, :);
%     solinit.y(2, :) = u_x(i, :);
%     sol = bvp4c(@bvpfcn, @bcfcn, solinit, options);
%     plot(sol.x, sol.y(1, :));
%     disp(i)
% end


%% No training
data = load("pinn_case_b_0.mat");
x = data.x_test';
u = data.u_pred;
u_x = data.u_x_pred;

figure;
hold on
for i = 1: 1000
    plot(x, u(i, :))
end

options = bvpset("RelTol", 1e-6, "Stats", "off");
solinit = bvpinit(x, @guess);

figure;
hold on
for i = 1: 1000
    solinit.y(1, :) = u(i, :);
    solinit.y(2, :) = u_x(i, :);
    sol = bvp4c(@bvpfcn, @bcfcn, solinit, options);
    if sol.stats.maxres > 0.001
        disp(i)
        disp(sol.stats.maxres)
        % plot(sol.x, sol.y(1, :), "-*");
    else
        plot(sol.x, sol.y(1, :));
    end
end


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
