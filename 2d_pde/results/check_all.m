clear all
close all
% 
% data = load("results3.mat");
% x_test = double(data.xx);
% y_test = double(data.yy);
% u_pred = double(data.uu);


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
                             "Edge",1:4, ...
                             "u",0);

specifyCoefficients( ...
    model, ...
    "m",0, ...
    "d",0, ...
    "c",-1, ...
    "a",0, ...
    "f",@f);
hmax = 0.05;
generateMesh(model, "Hmax", hmax);
% generateMesh(model, "Hmax", hmax)
figure
pdemesh(model); 
axis equal
xlim([0, 1])
ylim([0, 1])
xlabel("$x$", "FontSize", 20, "Interpreter", "latex")
ylabel("$y$", "FontSize", 20, "Interpreter", "latex")
ax = gca;
exportgraphics(ax,'mesh.png','Resolution',300) 


it10 = [0, 1, 2, 3, 4];
res10 = [5.5497e-01, 2.9405e-04, 1.6945e-07, 9.3374e-11, 9.8692e-14];
it9 = [0, 1, 2, 3, 4, 5, 6];
res9 = [6.5946e-01, 1.1259e-03, 1.6522e-06, 5.1701e-10, 2.9470e-13, 1.5363e-13, 9.7700e-14];
it8 = [0, 1, 2, 3, 4, 5];
res8 = [4.8455e-01, 5.2408e-04, 4.6773e-07, 2.2778e-10, 1.0481e-13, 9.1038e-14];
it7 = [0, 1, 2, 3, 4, 5, 6, 7, 8];
res7 = [3.9243e-01, 8.3614e-04, 6.4927e-07, 2.1214e-10, 1.4468e-13, 1.3096e-13, 1.2131e-13, 1.0658e-13, 8.5265e-14];
it6 = [0, 1, 2, 3, 4, 5];
res6 = [4.6553e-01, 5.6871e-03, 1.3105e-05, 9.5614e-09, 2.8495e-12, 9.9407e-14];
it5 = [0, 1, 2, 3, 4, 5];
res5 = [3.0106e-01, 9.7405e-05, 2.1720e-08, 2.5165e-11, 1.5270e-13, 9.9420e-14];
it4 = [0, 1, 2, 3, 4, 5, 6, 7];
res4 = [6.0562e-01, 1.1020e-02, 1.5276e-04, 1.9488e-07, 2.4053e-10, 1.8969e-12, 1.1517e-13, 8.7638e-14];
it3 = [0, 1, 2, 3, 4, 5, 6];
res3 = [2.8592e-01, 1.1873e-03, 5.0807e-07, 3.6669e-10, 2.8814e-13, 1.1727e-13, 9.8178e-14];
it2 = [0, 1, 2, 3];
res2 = [8.6800e-02, 2.0242e-06, 3.3663e-11, 6.2203e-14];
it1 = [0, 1, 2, 3, 4];
res1 = [2.9441e-01, 5.3980e-05, 2.6317e-09, 1.4161e-12, 8.6563e-14];

figure;
semilogy(it1, res1, "LineWidth", 2);
hold on;
semilogy(it2, res2, "LineWidth", 2);
semilogy(it3, res3, "LineWidth", 2);
semilogy(it4, res4, "LineWidth", 2);
semilogy(it5, res5, "LineWidth", 2);
semilogy(it6, res6, "LineWidth", 2);
semilogy(it7, res7, "LineWidth", 2);
semilogy(it8, res8, "LineWidth", 2);
semilogy(it9, res9, "LineWidth", 2);
semilogy(it10, res10, "LineWidth", 2);
xlabel("\# of iterations", "FontSize", 20, "Interpreter", "latex")
ylabel("Residual", "FontSize", 20, "Interpreter", "latex")
legend(["$u_1$", "$u_2$", "$u_3$", "$u_4$", "$u_5$", "$u_6$", "$u_7$", "$u_8$", "$u_9$", "$u_{10}$"], "FontSize", 20, "Interpreter", "latex")
legend boxoff
box on
ax = gca;
exportgraphics(ax,'residual.png','Resolution',300) 


function out = f(location, state)
out = -state.u .^ 2 + 800 * sin(pi*location.x) .* sin(pi*location.y);
end

function out = pinn(location)

x = location.x;
y = location.y;

%% Load parameters of NNs
data = load("params_case6_high.mat");
theta = data.theta;
idx = 293; % 24, 31, 47, 79
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