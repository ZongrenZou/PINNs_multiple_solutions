close all
clear all

data = load("pinn_case_b_all.mat");
x_test = data.x_test';
u = data.u_pred;
u_x = data.u_x_pred;

%% Check the value of u at x=-0.8 to choose initial guesses for each class
figure(1)
plot(u(:, 21), "k.")
idx1 = 362;
idx2 = 408;
idx3 = 417;
idx4 = 536;
idx5 = 509;
idx6 = 582;
idx7 = 784;
idx8 = 694;
idx9 = 612;

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
