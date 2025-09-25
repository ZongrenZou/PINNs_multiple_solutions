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


it10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
res10 = [7.0725e-01, 4.1623e-01, 3.1217e-01, 2.3615e-01, 1.9107e-01, 4.1441e-02, 3.3054e-04, 4.5058e-07, 3.1008e-10, 9.4542e-13, 1.5304e-13];
it9 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
res9 = [6.7894e-01, 4.9460e-01, 3.7095e-01, 3.1508e-01, 7.7239e-02, 1.2789e-03, 5.4584e-07, 6.0244e-10, 1.7579e-12, 1.2512e-13];
it8 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
res8 = [6.8163e-01, 3.6946e-01, 2.7919e-01, 2.4388e-01, 1.9521e-01, 4.0915e-02, 3.3320e-04, 2.4487e-07, 1.7337e-10, 1.1619e-12, 1.7812e-13];
it7 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
res7 = [6.8363e-01, 3.9262e-01, 2.8740e-01, 2.5351e-01, 2.0000e-01, 4.0410e-02, 3.4493e-04, 1.6059e-07, 3.2803e-10, 7.7875e-13, 1.5926e-13];
it6 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
res6 = [7.1320e-01, 4.8748e-01, 2.8275e-01, 1.3035e-01, 2.0538e-03, 1.7058e-06, 2.0698e-08, 6.9148e-11, 1.8374e-12, 1.4622e-13];
it5 = [0, 1, 2, 3, 4, 5, 6, 7];
res5 = [6.0562e-01, 1.1020e-02, 1.5276e-04, 1.9488e-07, 2.4053e-10, 1.8969e-12, 1.1517e-13, 8.7638e-14];
it4 = [0, 1, 2, 3, 4, 5, 6, 7, 8];
res4 = [7.0214e-01, 4.8190e-01, 2.6421e-01, 1.3272e-01, 1.7147e-03, 5.3454e-07, 7.0237e-10, 2.5136e-12, 1.4563e-13];
it3 = [0, 1, 2, 3, 4, 5, 6, 7, 8];
res3 = [6.7174e-01, 4.6778e-01, 2.6804e-01, 1.3248e-01, 1.7734e-03, 1.1142e-06, 8.9027e-09, 1.2088e-11, 1.4062e-13];
it2 = [0, 1, 2, 3, 4, 5, 6];
res2 = [6.8662e-01, 1.7332e-01, 4.3761e-03, 2.8366e-06, 5.5700e-09, 6.0912e-12, 5.6843e-14];
it1 = [0, 1, 2, 3, 4, 5, 6, 7];
res1 = [6.9466e-01, 2.4031e-01, 1.3558e-01, 4.6793e-04, 4.0650e-07, 2.5312e-10, 1.0554e-12, 1.8947e-13];

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
exportgraphics(ax,'residual_new.png','Resolution',300) 


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