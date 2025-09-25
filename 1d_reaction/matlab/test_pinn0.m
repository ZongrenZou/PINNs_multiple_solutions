close all
clear all
data = load("pinn_40.mat");
u = data.u_pred;
u_x = data.u_x_pred;
xmesh = linspace(-0.5, 0.5, 40);

% idx = (u > 10)

figure;
hold on
for i = 1: 1000
    plot(xmesh, u(:, i));
end

idx = u(25, :) < -1;
u_1 = u(:, idx);
u_x_1 = u_x(:, idx);
n1 = size(u_1);
n1 = n1(2);
figure;
hold on
solinit = bvpinit(xmesh, @guess1);
for i = 1: n1
    solinit.y(1, :) = u_1(:, i);
    solinit.y(2, :) = u_x_1(:, i);
    sol = bvp4c(@bvpfcn, @bcfcn, solinit);
    plot(sol.x, sol.y(1, :), "k-", "linewidth", 2)
    plot(xmesh, u_1(:, i), "r--", "linewidth", 2)
end


idx = u(75, :) > 1;
u_2 = u(:, idx);
u_x_2 = u_x(:, idx);
n2 = size(u_2);
n2 = n2(2);
figure;
hold on
solinit = bvpinit(xmesh, @guess1);
for i = 1: n2
    solinit.y(1, :) = u_2(:, i);
    solinit.y(2, :) = u_x_2(:, i);
    sol = bvp4c(@bvpfcn, @bcfcn, solinit);
    plot(sol.x, sol.y(1, :), "k-", "linewidth", 2)
    plot(xmesh, u_2(:, i), "r--", "linewidth", 2)
end


idx = (u(25, :) > 1) .* u(75, :) < -1;
u_3 = u(:, idx);
u_x_3 = u_x(:, idx);
n3 = size(u_3);
n3 = n3(2);
figure;
hold on
solinit = bvpinit(xmesh, @guess1);
for i = 1: n3
    solinit.y(1, :) = u_3(:, i);
    solinit.y(2, :) = u_x_3(:, i);
    sol = bvp4c(@bvpfcn, @bcfcn, solinit);
    plot(sol.x, sol.y(1, :), "k-", "linewidth", 2)
    plot(xmesh, u_3(:, i), "r--", "linewidth", 2)
end


disp([n1, n2, n3])



function dydx = bvpfcn(x, y) % equation to solve
dydx = [y(2)
        1/0.03 * (y(1) * y(2) - y(1))];
end
%--------------------------------

function res = bcfcn(ya, yb) % boundary conditions
res = [ya(1) - 1
       yb(1) + 1];
end
%--------------------------------

function g = guess1(x) % initial guess for y and y'
g = [sin(10*x)
     cos(10*x)];
end
