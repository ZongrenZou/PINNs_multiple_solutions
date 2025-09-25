import torch
from tools import *
import os
import cavity_data as cavity
import pinn_solver as psolver

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

        #filename = './nektarMesh.txt'
        #filename = './initialVelocityPoints.txt'
        filename = './initialPressurePoints.txt'
        x_star, y_star = dataloader.loading_evaluate_data(filename)
        # evaluate
        PINN.test(x_star, y_star)

if __name__ == "__main__":
    for loop in range(0,1):
        # 获取当前脚本文件所在的目录
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 将 net_params 设置为当前文件夹下 results 文件夹中的文件路径
        net_params = os.path.join(base_dir, 'results', 'Re1500_4x120_Nf20k_lamB10_loop2', 'checkpoint_epoch_800000.pth')
        train(net_params=net_params)
