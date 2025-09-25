clear all
close all

x = linspace(0, 1, 200);
y = linspace(0, 1, 200);
[xx, yy] = meshgrid(x, y);
xxx = reshape(xx, [40000, 1]);
yyy = reshape(yy, [40000, 1]);
location = struct();
location.x = xxx;
location.y = yyy;


%% Case 1
u_init = pinn(location, 1);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_1$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u1_init.png','Resolution',300) 


%% Case 2
u_init = pinn(location, 204);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_2$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u2_init.png','Resolution',300) 


%% Case 3
u_init = pinn(location, 5);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_3$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u3_init.png','Resolution',300)


%% Case 4
u_init = pinn(location, 26);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_4$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u4_init.png','Resolution',300)


%% Case 5
u_init = pinn(location, 19);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_5$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u5_init.png','Resolution',300)


%% Case 6
u_init = pinn(location, 51);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_6$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u6_init.png','Resolution',300)


%% Case 7
u_init = pinn(location, 378);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_7$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u7_init.png','Resolution',300)


%% Case 8
u_init = pinn(location, 148);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_8$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u8_init.png','Resolution',300)


%% Case 9
u_init = pinn(location, 199);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_9$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u9_init.png','Resolution',300)


%% Case 10
u_init = pinn(location, 293);
u_init = reshape(u_init, [200, 200]);

figure
pcolor(xx, yy, u_init)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u_{10}$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'u10_init.png','Resolution',300)



function out = pinn(location, idx)

x = location.x;
y = location.y;

%% Load parameters of NNs
data = load("params_case6_high.mat");
theta = data.theta;
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
out = out * 20;
% out = 1 * out;
% out = out .* (1 - x.^2 - y.^2);
% out = 1 * out + 0.00 * randn(size(out));

end

