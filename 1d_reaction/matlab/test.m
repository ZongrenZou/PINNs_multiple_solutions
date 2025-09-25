close all
clear all
xmesh = linspace(-0.5, 0.5, 1000);
%% R = 14


solinit1 = bvpinit(xmesh, @guess1);
sol1 = bvp4c(@bvpfcn, @bcfcn, solinit1);

solinit2 = bvpinit(xmesh, @guess2);
sol2 = bvp4c(@bvpfcn, @bcfcn, solinit2);
% 
% solinit3 = bvpinit(xmesh, @guess3);
% sol3 = bvp4c(@bvpfcn, @bcfcn, solinit3);


figure(1)
hold on
plot(sol1.x, sol1.y(1, :), 'k-', "linewidth", 2)
plot(sol2.x, sol2.y(1, :), 'r--', "linewidth", 2)


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

function g = guess2(x) % initial guess for y and y'
g = [10 * sin(5*x)
     10 * cos(5*x)];
end
