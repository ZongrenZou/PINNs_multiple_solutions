close all
clear all
data2 = load("reference.mat");
y1_ref = data2.y1;
y2_ref = data2.y2;
y3_ref = data2.y3;
x_ref = linspace(-0.5, 0.5, 150);

data = load("pinn0_new.mat");
u = data.u_pred;
u_x = data.u_x_pred;
xmesh = linspace(-0.5, 0.5, 2000);

% idx = (u > 10)

figure;
hold on
for i = 1: 1000
    plot(xmesh, u(:, i));
end

%% All

figure;
hold on
plot(x_ref, y1_ref, "k-", "linewidth", 2)
plot(x_ref, y2_ref, "k-", "linewidth", 2)
plot(x_ref, y3_ref, "k-", "linewidth", 2)
solinit = bvpinit(xmesh, @guess1);
for i = 1: 1000
    solinit.y(1, :) = u(:, i);
    solinit.y(2, :) = u_x(:, i);
    sol = bvp4c(@bvpfcn, @bcfcn, solinit);
    if sol.stats.maxres < 0.001
        plot(sol.x, sol.y(1, :), "r--", "linewidth", 2)
    end
    disp([i, sol.stats.maxres])
    % plot(xmesh, u_1(:, i), "r--", "linewidth", 2)
end
ylim([-2, 2])


%% the first mode
% % idx = u(10, :) < -1;
% % u_1 = u(:, idx);
% % u_x_1 = u_x(:, idx);
% % n1 = size(u_1);
% % n1 = n1(2);
% % figure;
% % hold on
% % plot(x_ref, y1_ref, "k-", "linewidth", 2)
% % plot(x_ref, y2_ref, "k-", "linewidth", 2)
% % plot(x_ref, y3_ref, "k-", "linewidth", 2)
% % solinit = bvpinit(xmesh, @guess1);
% % for i = 1: n1
% %     solinit.y(1, :) = u_1(:, i);
% %     solinit.y(2, :) = u_x_1(:, i);
% %     sol = bvp4c(@bvpfcn, @bcfcn, solinit);
% %     plot(sol.x, sol.y(1, :), "r--", "linewidth", 2)
% %     % plot(xmesh, u_1(:, i), "r--", "linewidth", 2)
% % end


%% the second mode
% idx = u(30, :) > 1;
% u_2 = u(:, idx);
% u_x_2 = u_x(:, idx);
% n2 = size(u_2);
% n2 = n2(2);
% figure;
% hold on
% plot(x_ref, y1_ref, "k-", "linewidth", 2)
% plot(x_ref, y2_ref, "k-", "linewidth", 2)
% plot(x_ref, y3_ref, "k-", "linewidth", 2)
% solinit = bvpinit(xmesh, @guess1);
% for i = 1: n2
%     solinit.y(1, :) = u_2(:, i);
%     solinit.y(2, :) = u_x_2(:, i);
%     sol = bvp4c(@bvpfcn, @bcfcn, solinit);
%     plot(sol.x, sol.y(1, :), "r--", "linewidth", 2)
%     % plot(xmesh, u_2(:, i), "r--", "linewidth", 2)
% end


%% the third mode
idx = (u(10, :) > 1) & (u(30, :) < -1);
u_3 = u(:, idx);
u_x_3 = u_x(:, idx);
n3 = size(u_3);
n3 = n3(2);
% figure;
% hold on
% plot(x_ref, y1_ref, "k-", "linewidth", 2)
% plot(x_ref, y2_ref, "k-", "linewidth", 2)
% plot(x_ref, y3_ref, "k-", "linewidth", 2)
% solinit = bvpinit(xmesh, @guess1);
% for i = 1: n3
%     solinit.y(1, :) = u_3(:, i);
%     solinit.y(2, :) = u_x_3(:, i);
%     sol = bvp4c(@bvpfcn, @bcfcn, solinit);
%     plot(sol.x, sol.y(1, :), "r--", "linewidth", 2)
%     % plot(xmesh, u_3(:, i), "r--", "linewidth", 2)
% end


%% the fourth mode
% idx = (u(1500, :) < 1) & (u(4500, :) > -1);
% u_4 = u(:, idx);
% u_x_4 = u_x(:, idx);
% n4 = size(u_4);
% n4 = n4(2);
% figure;
% hold on
% plot(x_ref, y1_ref, "k-", "linewidth", 2)
% plot(x_ref, y2_ref, "k-", "linewidth", 2)
% plot(x_ref, y3_ref, "k-", "linewidth", 2)
% solinit = bvpinit(xmesh, @guess1);
% % disp([n1, n2, n3, n4])
% 
% for i = 1: n4
%     solinit.y(1, :) = u_4(:, i);
%     solinit.y(2, :) = u_x_4(:, i);
%     sol = bvp4c(@bvpfcn, @bcfcn, solinit);
%     if sol.stats.maxres < 1
%         disp(i)
%         plot(sol.x, sol.y(1, :), "r--", "linewidth", 2)
%     end
% 
%     % plot(xmesh, u_3(:, i), "r--", "linewidth", 2)
% end
% ylim([-2, 2])




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
