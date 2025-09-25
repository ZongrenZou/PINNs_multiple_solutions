clear all
close all



it1 = [0, 1, 2, 3, 4, 5];
res1 = [1.3575e+00, 1.9349e-06, 4.4032e-10, 2.2417e-13, 1.1049e-15, 2.2894e-17];

it2 = [0, 1, 2, 3, 4];
res2 = [1.2039e+00, 6.8552e-07, 1.3516e-10, 3.3990e-14, 3.8205e-17];

it3 = [0, 1, 2, 3];
res3 = [1.4809e+00, 5.4517e-06, 1.0326e-11, 4.6090e-16];



figure;
semilogy(it1, res1, "LineWidth", 2);
hold on;
semilogy(it2, res2, "LineWidth", 2);
semilogy(it3, res3, "LineWidth", 2);
xlabel("\# of iterations", "FontSize", 20, "Interpreter", "latex")
ylabel("Residual", "FontSize", 20, "Interpreter", "latex")
legend(["$u_1$", "$u_2$", "$u_3$"], "FontSize", 20, "Interpreter", "latex")
legend boxoff
box on
ax = gca;
ax.XTick = unique( round(ax.XTick) );
exportgraphics(ax,'residual.png','Resolution',300) 