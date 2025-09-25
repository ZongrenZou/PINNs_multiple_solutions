# import torch
# from tools import *
import cavity_data as cavity
import pinn_solver as psolver
import time
import scipy

def train(net_params=None, loop=None):
        Re = 1500 # Reynolds number
        layers = 4
        hidden_size = 120
        layers_1 = 4
        hidden_size_1 = 30
        lam_bcs = 10
        lam_equ = 1
        N_f = 20000
        N_b = 1000
        loop=loop
    
        PINN = psolver.PysicsInformedNeuralNetwork(
            Re=Re,
            layers=layers,
            hidden_size=hidden_size,
            layers_1=layers_1,
            hidden_size_1=hidden_size_1,
            N_f = N_f,
            loop=loop,
            bc_weight=lam_bcs,
            eq_weight=lam_equ,
            net_params=net_params,
            checkpoint_path='./checkpoint/')
    
        path = './datasets/'
        dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=N_b)
    
        # Set boundary data, | u, v, x, y
        boundary_data = dataloader.loading_boundary_data()
        PINN.set_boundary_data(X=boundary_data)
    
        # Set training data, | x, y
        training_data = dataloader.loading_training_data()
        PINN.set_eq_training_data(X=training_data)
        # Set test data,  | u, v, p, x, y
        filename = './data/cavity_Re'+str(Re)+'_257.mat'
        x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(filename)

        # Training
        start_time = time.time()
        epoch= 200000
        PINN.set_alpha_evm(0.05)
        PINN.train(num_epoch=epoch, lr=1e-3, label=1*epoch, x_test=x_star, y_test=y_star, u_test=u_star, v_test=v_star)
        PINN.test(x_star, y_star, u_star, v_star, p_star, 1*epoch, loop)

        PINN.set_alpha_evm(0.03)
        PINN.train(num_epoch=epoch, lr=2e-4, label=2*epoch, x_test=x_star, y_test=y_star, u_test=u_star, v_test=v_star)
        PINN.test(x_star, y_star, u_star, v_star, p_star, 2*epoch, loop)

        PINN.set_alpha_evm(0.02)
        PINN.train(num_epoch=epoch, lr=1e-4, label=3*epoch, x_test=x_star, y_test=y_star, u_test=u_star, v_test=v_star)
        PINN.test(x_star, y_star, u_star, v_star, p_star, 3*epoch, loop)

        PINN.set_alpha_evm(0.01)
        PINN.train(num_epoch=epoch, lr=5e-5, label=4*epoch, x_test=x_star, y_test=y_star, u_test=u_star, v_test=v_star)
        PINN.test(x_star, y_star, u_star, v_star, p_star, 4*epoch, loop)

        PINN.save_summary(loop)
        train_time = time.time() - start_time
        print(train_time)
        filename = f'training_time{loop}.mat'
        scipy.io.savemat(filename, {'train_time': train_time})

if __name__ == "__main__":
    for loop in range(0,5):
        train(loop=loop)
