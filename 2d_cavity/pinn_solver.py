import os
import scipy.io
import numpy as np
from net import FCNet
#from torchinfo import summary
import matplotlib.pyplot as plt
import torch
from typing import List, Optional


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PysicsInformedNeuralNetwork:
    # Initialize the class
    # training_type:  'unsupervised' | 'half-supervised'
    def __init__(self,
                 opt=None,
                 Re = 2000,
                 layers=6,
                 hidden_size=80,
                 layers_1=6,
                 hidden_size_1=80,
                 N_f = 4000,
                 alpha_evm=0.03,
                 loop=1,
                 learning_rate=0.001,
                 outlet_weight=1,
                 bc_weight=10,
                 eq_weight=1,
                 ic_weight=1,
                 num_ins=2,  #输入x，y
                 num_outs=3,  #输出u、v、p
                 num_outs_1=1,  # 输出u、v、p
                 supervised_data_weight=1,
                 training_type='unsupervised',  #无监督学习
                 net_params=None,
                 net_params_1=None,
                 checkpoint_freq=2000,
                 checkpoint_path='./checkpoint/'):
        
        self.Re = Re #雷诺数
        self.N_f = N_f
        self.loop = loop
        self.vis_e0 = 5.0/self.Re

        self.layers = layers
        self.hidden_size = hidden_size
        self.layers_1 = layers_1
        self.hidden_size_1 = hidden_size_1

        self.alpha_evm = alpha_evm
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.training_type = training_type
        self.alpha_b = bc_weight
        self.alpha_e = eq_weight
        self.alpha_i = ic_weight
        self.alpha_o = outlet_weight
        self.alpha_s = supervised_data_weight
        self.loss_i = self.loss_o = self.loss_b = self.loss_e = self.loss_s = 0.0
        self.loss_bcs_all = []   #用于记录loss
        self.loss_equ_all = []
        self.loss_sum_all = []
        self.loss_e1_all = []
        self.loss_e2_all = []
        self.loss_e3_all = []
        self.all_loss = 0
        self.eq_loss = 0
        self.bc_loss = 0
        self.e1_loss = 0
        self.error_u_all = []
        self.error_v_all = []
        # initialize NN
        self.net = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs, num_layers=layers, hidden_size=hidden_size).to(device)

        self.net_1 = self.initialize_NN(
                num_ins=num_ins, num_outs=num_outs_1, num_layers=layers_1, hidden_size=hidden_size_1).to(device)

        if net_params:
            load_params = torch.load(net_params, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),weights_only=True)
            self.net.load_state_dict(load_params['model_state_dict'])

        if net_params_1:
            load_params_1 = torch.load(net_params_1, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),weights_only=True)
            self.net.load_state_dict(load_params_1['model_state_dict_1'])

        self.opt = torch.optim.Adam(
            list(self.net.parameters())+list(self.net_1.parameters()),
            lr=learning_rate,
            weight_decay=0.0) if not opt else opt    #使用adam优化器

    def prepare_tensor(self, tensor, requires_grad):
        """
        辅助函数：将张量转换为双精度并移动到指定设备。

        参数:
        - tensor (torch.Tensor 或 numpy.ndarray): 输入张量。
        - requires_grad (bool): 是否需要计算梯度。

        返回:
        - torch.Tensor: 处理后的张量。
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError("输入必须是 torch.Tensor 或 numpy.ndarray。")

        return tensor.clone().detach().double().to(self.device).requires_grad_(requires_grad)
    def init_vis_t(self):
        """
        初始化人工粘性相关参数。

        """
        (_, _, _, e) = self.neural_net_u(self.x_f, self.y_f)
        self.vis_e_minus  = self.alpha_evm*torch.abs(e).detach().cpu().numpy()

    def set_boundary_data(self, X=None):
        # boundary training data | u, v, t, x, y
        requires_grad = False #不需要计算梯度
        # 将边界数据转换为 PyTorch 张量并移动到设备上
        self.x_b = self.prepare_tensor(X[0], requires_grad)
        self.y_b = self.prepare_tensor(X[1], requires_grad)
        self.u_b = self.prepare_tensor(X[2], requires_grad)
        self.v_b = self.prepare_tensor(X[3], requires_grad)

    def set_eq_training_data(self,X=None):
        # inferior training data | u, v, t, x,
        requires_grad = True #需要计算梯度
        # 将方程数据转换为 PyTorch 张量并移动到设备上
        self.x_f = self.prepare_tensor(X[0], requires_grad)
        self.y_f = self.prepare_tensor(X[1], requires_grad)
        self.init_vis_t()

    def set_optimizers(self, opt):
        self.opt = opt #优化器

    def set_alpha_evm(self, alpha):
        self.alpha_evm = alpha


    def initialize_NN(self,
                      num_ins=2,
                      num_outs=3,
                      num_layers=10,
                      hidden_size=50):
        return FCNet(num_ins=num_ins,
                     num_outs=num_outs,
                     num_layers=num_layers,
                     hidden_size=hidden_size,
                     activation=torch.nn.Tanh)

    def neural_net_u(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvp = self.net(X)
        ee = self.net_1(X)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        e = ee[:, 0:1]
        return u, v, p, e

    def neural_net_equations(self, x, y):
        X = torch.cat((x, y), dim=1)
        uvp = self.net(X)
        ee = self.net_1(X)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        e = ee[:, 0:1]

        u_x, u_y = self.autograd(u, [x,y])
        u_xx = self.autograd(u_x, [x])[0]
        u_yy = self.autograd(u_y, [y])[0]

        v_x, v_y = self.autograd(v, [x,y])
        v_xx = self.autograd(v_x, [x])[0]
        v_yy = self.autograd(v_y, [y])[0]

        p_x, p_y = self.autograd(p, [x,y]) #计算u，v，p，t的梯度

        # Get the minum between (vis_t0, vis_t_mius(calculated with last step e))
        self.vis_t =   self.prepare_tensor(np.minimum(self.vis_e0, self.vis_e_minus), requires_grad=False)
        # Save vis_t_minus for computing vis_t in the next step
        self.vis_e_minus  = self.alpha_evm*torch.abs(e).detach().cpu().numpy()

        #构建NS方程和能量方程
        eq1 = (u*u_x + v*u_y) + p_x - (1.0/self.Re+self.vis_t)*(u_xx + u_yy)
        eq2 = (u*v_x + v*v_y) + p_y - (1.0/self.Re+self.vis_t)*(v_xx + v_yy)
        eq3 = u_x + v_y

        residual = (eq1*(u-0.5)+eq2*(v-0.5))-e

        return eq1, eq2, eq3, residual

    @torch.jit.script  #计算梯度
    def autograd(y: torch.Tensor, x: List[torch.Tensor]) -> List[torch.Tensor]:

        if not y.is_floating_point() or y.dtype != torch.float64:
            raise TypeError("y 必须是双精度浮点张量（torch.float64）。")

        for xi in x:
            if not xi.is_floating_point() or xi.dtype != torch.float64:
                raise TypeError("所有输入张量 x 必须是双精度浮点张量（torch.float64）。")

        # 创建与 y 相同 dtype 的 grad_outputs
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y, dtype=torch.float64, device=y.device)]

        grad = torch.autograd.grad(
            [y],
            x,
            grad_outputs=grad_outputs,
            create_graph=True,
            allow_unused=True,
            # retain_graph=True,  # 根据需要启用
        )

        # 处理可能的 None 梯度
        if grad is None:
            grad = [torch.zeros_like(xx, dtype=torch.float64, device=xx.device) for xx in x]
        else:
            grad = [g if g is not None else torch.zeros_like(xi, dtype=torch.float64, device=xi.device) for xi, g in zip(x, grad)]
        return grad

    def predict(self, net_params, X):
        x, y = X
        return self.neural_net_u(x, y)

    def shuffle(self, tensor):
        tensor_to_numpy = tensor.detach().cpu()
        # shuffle_numpy = np.random.shuffle(tensor_to_numpy)
        return torch.tensor(tensor_to_numpy, requires_grad=True).float()

    def fwd_computing_loss_2d(self, loss_mode='MSE'):
        # boundary data
        (self.u_pred_b, self.v_pred_b, _, _) = self.neural_net_u(self.x_b, self.y_b)

        # BC loss 边界的损失
        if loss_mode == 'L2':
            self.loss_b = torch.norm((self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1])), p=2) + \
                          torch.norm((self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])), p=2)
        if loss_mode == 'MSE':
            self.loss_b = torch.mean(torch.square(self.u_b.reshape([-1]) - self.u_pred_b.reshape([-1]))) + \
                          torch.mean(torch.square(self.v_b.reshape([-1]) - self.v_pred_b.reshape([-1])))

        # equation 方程的损失
        assert self.x_f is not None and self.y_f is not None

        (self.eq1_pred, self.eq2_pred, self.eq3_pred, self.eq4_pred) = self.neural_net_equations(self.x_f, self.y_f)
        
        if loss_mode == 'L2':
            self.loss_e = torch.norm(self.eq1_pred.reshape([-1]), p=2) + \
                  torch.norm(self.eq2_pred.reshape([-1]), p=2) + \
                  torch.norm(self.eq3_pred.reshape([-1]), p=2)
        if loss_mode == 'MSE':
            self.loss_eq1 = torch.mean(torch.square(self.eq1_pred.reshape([-1])))
            self.loss_eq2 = torch.mean(torch.square(self.eq2_pred.reshape([-1])))
            self.loss_eq3 = torch.mean(torch.square(self.eq3_pred.reshape([-1])))
            self.loss_eq4 = torch.mean(torch.square(self.eq4_pred.reshape([-1])))
            self.loss_e = self.loss_eq1+self.loss_eq2+self.loss_eq3 + 0.1*self.loss_eq4
        # 总损失
        self.loss = self.alpha_b * self.loss_b + self.alpha_e * self.loss_e

        return self.loss, [self.loss_e, self.loss_b]

    def train(self,
              num_epoch=1,
              lr=1e-4,
              label=None,
              optimizer=None,
              scheduler=None,
              batchsize=None,
              x_test = None,
              y_test = None,
              u_test = None,
              v_test = None):
        if self.opt is not None:
            self.opt.param_groups[0]['lr'] = lr
        else:
            self.opt = torch.optim.Adam(list(self.net.parameters()), lr=lr)
        return self.solve_Adam(self.fwd_computing_loss_2d, num_epoch, label, batchsize, scheduler, x_test, y_test, u_test, v_test)

    def solve_Adam(self,
                   loss_func,
                   num_epoch = 1000,
                   label = None,
                   batchsize = None,
                   scheduler = None,
                   x_test = None,
                   y_test = None,
                   u_test = None,
                   v_test = None):
        self.freeze_evm_net(0)
        for epoch_id in range(num_epoch):
            if epoch_id !=0 and epoch_id % 20000 == 0:
                self.defreeze_evm_net(epoch_id)
            if (epoch_id - 1) % 20000 == 0:
                self.freeze_evm_net(epoch_id)
            loss, losses = loss_func()  # 计算损失
            loss.backward()  # 反向传播计算梯度
            self.opt.step()  # 更新模型参数
            self.opt.zero_grad()  # 梯度清零

            # 提取并记录各种损失值
            e_loss = losses[0].detach().cpu().item()
            all_loss = loss.detach().cpu().item()
            bc_loss = losses[1].detach().cpu().item()
            e1_loss = self.loss_eq1.detach().cpu().item()
            e2_loss = self.loss_eq2.detach().cpu().item()
            e3_loss = self.loss_eq3.detach().cpu().item()

            self.loss_bcs_all.append([bc_loss])
            self.loss_equ_all.append([e_loss])
            self.loss_sum_all.append([all_loss])
            self.loss_e1_all.append([e1_loss])
            self.loss_e2_all.append([e2_loss])
            self.loss_e3_all.append([e3_loss])

            if scheduler:
                scheduler.step()  # 更新学习率

            # 打印日志
            if epoch_id == 0 or (epoch_id + 1) % 100 == 0:
                self.print_log(loss, losses, epoch_id, num_epoch)

            if (epoch_id + 1) % 100 == 0:
                self.error(x_test, y_test, u_test, v_test)

            if (epoch_id + 1) % 50000 == 0:
                self.opt.state.clear()
                self.save_checkpoint(epoch_id, label, N_HLayer=self.layers, N_neu=self.hidden_size, N_f=self.N_f)
    def freeze_evm_net(self, epoch_id):
        for para in self.net_1.parameters():
            para.requires_grad = False
        self.opt.param_groups[0]['params'] = list(self.net.parameters())

    def defreeze_evm_net(self, epoch_id):
        for para in self.net_1.parameters():
            para.requires_grad = True
        self.opt.param_groups[0]['params'] = list(self.net.parameters())+list(self.net_1.parameters())

    def print_log(self, loss, losses, epoch_id, num_epoch):

        # 获取当前学习率
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        # 打印当前学习率
        print("current lr is {}".format(get_lr(self.opt)))
        if isinstance(losses[0], int):
            eq_loss = losses[0]
        else:
            eq_loss = losses[0].detach().cpu().item()
        # 打印当前轮次信息和损失信息
        print("epoch/num_epoch: ", epoch_id + 1, "/", num_epoch,
              "loss[Adam]: %.3e"%(loss.detach().cpu().item()),
              "eq_loss: %.3e" %(losses[0].detach().cpu().item()),
              "bc_loss: %.3e" %(losses[1].detach().cpu().item()))

    def error(self, x_test, y_test, u_test, v_test):
        # Prediction
        x_test = self.prepare_tensor(x_test,requires_grad=False)
        y_test = self.prepare_tensor(y_test,requires_grad=False)
        u_pred, v_pred, p_pred,_ = self.neural_net_u(x_test, y_test)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1, 1)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1, 1)
        # Error
        error_u = np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(u_test, 2)
        error_v = np.linalg.norm(v_test - v_pred, 2) / np.linalg.norm(v_test, 2)

        self.error_u_all.append([error_u])
        self.error_v_all.append([error_v])

    def evaluate(self, x, y, u, v):
        """ testing all points in the domain """
        x_test = x.reshape(-1, 1)
        y_test = y.reshape(-1, 1)
        u_test = u.reshape(-1, 1)
        v_test = v.reshape(-1, 1)
        num_elements = x_test.size
        sqrt_num = np.sqrt(num_elements)
        # 打印结果
        # print('x_test的元素个数:', num_elements)
        # print('元素个数的平方根:', sqrt_num)
        # Prediction
        x_test = self.prepare_tensor(x_test, requires_grad=False)
        y_test = self.prepare_tensor(y_test, requires_grad=False)
        u_pred, v_pred, p_pred,_ = self.neural_net_u(x_test, y_test)
        u_pred = u_pred.detach().cpu().numpy().reshape(-1, 1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1, 1)
        # Error
        error_u = np.linalg.norm(u_test - u_pred, 2) / np.linalg.norm(u_test, 2)
        error_v = np.linalg.norm(v_test - v_pred, 2) / np.linalg.norm(v_test, 2)
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('------------------------')
        # plot picture

        error_u = np.abs(u_test - u_pred).reshape(int(sqrt_num), int(sqrt_num))
        error_v = np.abs(v_test - v_pred).reshape(int(sqrt_num), int(sqrt_num))
        u_test = u_test.reshape(257, 257)
        v_test = v_test.reshape(257, 257)
        u_pred = u_pred.reshape(257,257)
        v_pred = v_pred.reshape(257,257)
        x_test = x_test.cpu().numpy().reshape(int(sqrt_num), int(sqrt_num))
        y_test = y_test.cpu().numpy().reshape(int(sqrt_num), int(sqrt_num))

        plt.figure(figsize=(25, 25))
        plt.subplot(3, 3, 1)
        plt.pcolormesh(x_test, y_test, u_test, shading='auto',cmap='jet')
        plt.colorbar()
        plt.title('Reference_U')

        plt.subplot(3, 3, 2)
        plt.pcolormesh(x_test, y_test, u_pred, shading='auto',cmap='jet')
        plt.colorbar()
        plt.title('Pred_PINN_U')

        plt.subplot(3, 3, 3)
        plt.pcolormesh(x_test, y_test, error_u, shading='auto',cmap='jet')
        plt.colorbar()
        plt.title('Error_U')

        plt.subplot(3, 3, 4)
        plt.pcolormesh(x_test, y_test, v_test, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title('Reference_V')

        plt.subplot(3, 3, 5)
        plt.pcolormesh(x_test, y_test, v_pred, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title('Pred_PINN_V')

        plt.subplot(3, 3, 6)
        plt.pcolormesh(x_test, y_test, error_v, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title('Error_V')

        plt.show()
        plt.savefig('result_plot.png')
        plt.close()

    def test(self, x, y, u, v, p, num_epoch, loop=None):
        # 将输入数据重新形状为二维张量
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        u_test = u.reshape(-1,1)
        v_test = v.reshape(-1,1)

        # 将测试数据转换为张量并移动到设备上
        x_test = self.prepare_tensor(x_test, requires_grad=False)
        y_test = self.prepare_tensor(y_test, requires_grad=False)
        # 使用神经网络进行预测
        u_pred, v_pred, p_pred,_ = self.neural_net_u(x_test, y_test)
        # 将预测结果转换为 NumPy 数组
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        p_pred = p_pred.detach().cpu().numpy().reshape(-1,1)
        # Error
        # 计算误差
        error_u = np.linalg.norm(u_test-u_pred, 2)/np.linalg.norm(u_test, 2)
        error_v = np.linalg.norm(v_test-v_pred, 2)/np.linalg.norm(v_test, 2)
        # 打印误差信息
        print('------------------------')
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('------------------------')
        # 将预测结果重新形状为二维数组
        u_pred = u_pred.reshape(257,257)
        v_pred = v_pred.reshape(257,257)
        p_pred = p_pred.reshape(257,257)
        
        result_folder = 'result_data'
        os.makedirs(result_folder, exist_ok=True)
        
        # 保存.mat文件到新建文件夹中
        mat_file_path = os.path.join(result_folder, 'cavity_result_loop_%d_epoch%d.mat' % (loop, num_epoch))
        scipy.io.savemat(mat_file_path,
                    {'Error_u':error_u,
                     'Error_v':error_v,
                     'U_pred':u_pred,
                     'V_pred':v_pred,
                     'P_pred':p_pred,
                     'lam_bcs':self.alpha_b,
                     'lam_equ':self.alpha_e,
                     'loss_bcs_all':self.loss_bcs_all,
                     'loss_equ_all':self.loss_equ_all,
                     'loss_sum_all':self.loss_sum_all,
                     'error_u_all': self.error_u_all,
                     'error_v_all': self.error_v_all})

    def test(self, x, y):
        # 将输入数据重新形状为二维张量
        x_test = x.reshape(-1,1)
        y_test = y.reshape(-1,1)
        # 将测试数据转换为张量并移动到设备上
        x_test_torch = self.prepare_tensor(x_test, requires_grad=False)
        y_test_torch = self.prepare_tensor(y_test, requires_grad=False)
        # 使用神经网络进行预测
        u_pred, v_pred, p_pred,_ = self.neural_net_u(x_test_torch, y_test_torch)
        # 将预测结果转换为 NumPy 数组
        u_pred = u_pred.detach().cpu().numpy().reshape(-1,1)
        v_pred = v_pred.detach().cpu().numpy().reshape(-1,1)
        p_pred = p_pred.detach().cpu().numpy().reshape(-1,1)

        #np.savetxt('Case1.txt', np.column_stack([x_test,y_test,u_pred,v_pred,p_pred]), delimiter=' ',newline=' ') 
        #np.savetxt('Case5.txt', np.c_[x_test,y_test,u_pred,v_pred,p_pred], fmt='%.18g') 
        #np.savetxt('DealIICase3_velocity.txt', np.c_[x_test,y_test,u_pred,v_pred,p_pred], fmt='%.18g') 
        np.savetxt('DealIICase3_pressure.txt', np.c_[x_test,y_test,u_pred,v_pred,p_pred], fmt='%.18g') 
        
    def save_summary(self, loop):
        filename = f"model_summary_{loop}.txt"

        with open(filename, "w") as f:
            # 确保引用正确的模型
            # 根据模型的输入特征数量调整 input_size
            summary_str = summary(
                self.net,       # 确保这里引用的是正确的模型
                input_size=(1, 2),       # 根据模型的输入特征数量调整
                verbose=0,
                dtypes=[torch.float64]     # 指定输入数据类型为双精度
            )
            f.write(str(summary_str))

    def save_checkpoint(self, epoch_id, label, N_HLayer, N_neu, N_f, directory=None):

        # 设置保存目录为当前文件目录下的 results 文件夹
        if not directory:
            # 获取当前脚本所在的文件夹路径，并创建 results 文件夹
            base_dir = os.path.dirname(os.path.abspath(__file__))
            directory = os.path.join(base_dir, "results")

        # 检查或创建 results 目录
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 构建文件夹结构
        Re_folder = 'Re' + str(self.Re)
        NNsize = f"{N_HLayer}x{N_neu}_Nf{int(N_f / 1000)}k"
        lambdas = 'lamB' + str(self.alpha_b)
        loop_num = 'loop' + str(self.loop)
        relative_path = f"{Re_folder}_{NNsize}_{lambdas}_{loop_num}/"

        # 完整的保存路径
        save_results_to = os.path.join(directory, relative_path)

        # 创建完整的结果路径（如果不存在）
        if not os.path.exists(save_results_to):
            os.makedirs(save_results_to)

        # 生成带 epoch 的文件名并保存模型状态
        filename = f"checkpoint_epoch_{label}.pth"
        filepath = os.path.join(save_results_to, filename)

        print(f"Saving checkpoint to: {filepath}")  # 调试输出

        torch.save({
            'epoch': epoch_id,
            'model_state_dict': self.net.state_dict(),
            # 假设 self.opt 是优化器
            'optimizer_state_dict': self.opt.state_dict(),
            # 假设 loss 是计算的损失值
            'loss': self.loss
        }, filepath)
