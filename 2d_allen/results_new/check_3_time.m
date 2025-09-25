clear all
close all



%% Step 1: 3e-4 tol
model = createpde();
% Coordinates
lowerLeft  = [0   ,0   ];
lowerRight = [1 , 0  ];
upperRight = [1 , 1];
upperLeft =  [0.0 , 1];
% Geometry matrix
S = [3,4 lowerLeft(1), lowerRight(1), upperRight(1), upperLeft(1), ...
         lowerLeft(2), lowerRight(2), upperRight(2), upperLeft(2)];                     
gdm = S';
% Names
ns = 'S';
% Set formula 
sf = 'S';
% Invoke decsg
g = decsg(gdm,ns,sf');

geometryFromEdges(model,g);
% pdegplot(model,'EdgeLabel','on');

applyBoundaryCondition(model,"dirichlet", ...
                             "Edge",1, ...
                             "u",-1);
applyBoundaryCondition(model,"dirichlet", ...
                             "Edge",3, ...
                             "u",-1);
applyBoundaryCondition(model,"dirichlet", ...
                             "Edge",2, ...
                             "u",1);
applyBoundaryCondition(model,"dirichlet", ...
                             "Edge",4, ...
                             "u",1);

specifyCoefficients( ...
    model, ...
    "m",0, ...
    "d",1, ...
    "c",0.01, ...
    "a",0, ...
    "f",@f);
hmax = 0.05;
generateMesh(model, Hmax=hmax, Hvertex={[1, 2, 3, 4], 0.0001});
% generateMesh(model, "Hmax", hmax)
figure
pdemesh(model); 
axis equal

setInitialConditions(model, @pinn);


% model.SolverOptions.ResidualTolerance = 1.e-15;
model.SolverOptions.ReportStatistics = 'on';
model.SolverOptions.AbsoluteTolerance = 1e-12;
model.SolverOptions.RelativeTolerance = 1e-12;
% model.SolverOptions.MaxIterations = 500;
% model.SolverOptions.MinStep = eps;
tlist = [0, 5, 10, 20, 50];

results = solvepde(model, tlist);


x = linspace(0, 1, 200);
y = linspace(0, 1, 200);
[xx, yy] = meshgrid(x, y);
xxx = reshape(xx, [40000, 1]);
yyy = reshape(yy, [40000, 1]);
location = struct();
location.x = xxx;
location.y = yyy;
u_init = pinn(location);
u_init = reshape(u_init, [200, 200]);

u0 = interpolateSolution(results, xx, yy, 1);
u0 = reshape(u0, [200, 200]);

u1 = interpolateSolution(results, xx, yy, 2);
u1 = reshape(u1, [200, 200]);
u2 = interpolateSolution(results, xx, yy, 3);
u2 = reshape(u2, [200, 200]);
u3 = interpolateSolution(results, xx, yy, 4);
u3 = reshape(u3, [200, 200]);
u4 = interpolateSolution(results, xx, yy, 5);
u4 = reshape(u4, [200, 200]);





figure
pcolor(xx, yy, u0)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u|_{t=0}$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'case3_0.png','Resolution',300) 


figure
pcolor(xx, yy, u1)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u|_{t=5}$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'case3_1.png','Resolution',300) 


figure
pcolor(xx, yy, u2)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u|_{t=10}$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'case3_2.png','Resolution',300) 


figure
pcolor(xx, yy, u3)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u|_{t=20}$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'case3_3.png','Resolution',300) 


figure
pcolor(xx, yy, u4)
colormap(jet)
shading interp;
colorbar
axis("equal")
xlim([0, 1])
ylim([0, 1])
title("$u|_{t=50}$", "FontSize", 20, "Interpreter", "latex")
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'case3_4.png','Resolution',300) 



function out = f(location, state)
out = -state.u .^ 3 + state.u;
end

function out = pinn(location)

x = location.x;
y = location.y;

%% Load parameters of NNs
data = load("params_case2.mat");
theta = data.theta;
idx = 6; % 24, 31, 47, 79
theta = double(theta(idx, :));

st = 1; ed = 100;
w1 = reshape(theta(:, st:ed), [50, 2])';
st = ed + 1; ed = ed + 50*50;
w2 = reshape(theta(:, st:ed), [50, 50])';
st = ed + 1; ed = ed + 50*50;
w3 = reshape(theta(:, st:ed), [50, 50])';
st = ed + 1; ed = ed + 50*1;
w4 = reshape(theta(:, st:ed), [50, 1]);
st = ed + 1; ed = ed + 1*50;
b1 = reshape(theta(:, st:ed), [1, 50]);
st = ed + 1; ed = ed + 1*50;
b2 = reshape(theta(:, st:ed), [1, 50]);
st = ed + 1; ed = ed + 1*50;
b3 = reshape(theta(:, st:ed), [1, 50]);
st = ed + 1; ed = ed + 1*1;
b4 = reshape(theta(:, st:ed), [1, 1]);

x = reshape(x, [length(x), 1]);
y = reshape(y, [length(y), 1]);

out = [x, y];
out = tanh(out * w1 + b1);
out = tanh(out * w2 + b2);
out = tanh(out * w3 + b3);
out = out * w4 + b4;
% out = 1 * out;
% out = out .* (1 - x.^2 - y.^2);
% out = 1 * out + 0.00 * randn(size(out));

end

