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
    "d",0, ...
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


model.SolverOptions.ResidualTolerance = 1.e-15;
model.SolverOptions.ReportStatistics = 'on';
model.SolverOptions.MaxIterations = 500;
model.SolverOptions.MinStep = eps;

results = solvepde(model);


x = linspace(0, 1, 200);
y = linspace(0, 1, 200);
[xx, yy] = meshgrid(x, y);
u = interpolateSolution(results, xx, yy);
u = reshape(u, [200, 200]);
xxx = reshape(xx, [40000, 1]);
yyy = reshape(yy, [40000, 1]);
location = struct();
location.x = xxx;
location.y = yyy;
u_init = pinn(location);
u_init = reshape(u_init, [200, 200]);


figure
pcolor(xx, yy, u)
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
exportgraphics(ax,'u3.png','Resolution',300) 

% figure
% surf(xx, yy, u_init);
% % pcolor(xx, yy, u)
% colormap(jet)
% shading interp;
% colorbar
% % contour(xx, yy, u, "LineWidth", 1)
% % axis("equal")
% title("A PINN solution")
% % clim([0, 2])
% xlabel("x")
% ylabel("y")





function out = f(location, state)
out = -state.u .^ 3 + state.u;
end

function out = pinn(location)

x = location.x;
y = location.y;

%% Load parameters of NNs
data = load("params_case2.mat");
theta = data.theta;
idx = 31; % 24, 31, 47, 79
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

