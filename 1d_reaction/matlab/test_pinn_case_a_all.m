close all
clear all

data = load("pinn_case_a_all.mat");
x_test = data.x_test';
u = data.u_pred;
u_x = data.u_x_pred;

%% Check the value of u at x=-0.8 to choose initial guesses for each class
figure(1)
plot(u(:, 21), "k.")
idx1 = 344;
idx2 = 294;
idx3 = 397;
idx4 = 606;
idx5 = 409;
idx6 = 285;
idx7 = 539;
idx8 = 516;
idx9 = 369;
idx10 = 371;
idx11 = 321;

figure
hold on
plot(x_test, u(idx1, :, :));
plot(x_test, u(idx2, :, :));
plot(x_test, u(idx3, :, :));
plot(x_test, u(idx4, :, :));
plot(x_test, u(idx5, :, :));
plot(x_test, u(idx6, :, :));
plot(x_test, u(idx7, :, :));
plot(x_test, u(idx8, :, :));
plot(x_test, u(idx9, :, :));
plot(x_test, u(idx10, :, :));
plot(x_test, u(idx11, :, :));

