import numpy as np
import scipy.io
from tools import LHSample
from tools import sort_pts
# from scipy.stats import qmc


class DataLoader:
    def __init__(self, path=None, N_f=None, N_b=None):

        '''
        N_f: Num of residual points
        N_b: Num of boundary points
        '''
        # 将传入的边界点数量参数赋值给实例变量 self.N_b
        # 将传入的边界点数量参数赋值给实例变量 self.N_b
        self.N_b = N_b
        # 设置 x 轴的最小值为 0.0
        self.x_min = 0.0
        # 设置 x 轴的最大值为 1.0
        self.x_max = 1.0
        # 设置 y 轴的最小值为 0.0
        self.y_min = 0.0
        # 设置 y 轴的最大值为 1.0
        self.y_max = 1.0
        # 将传入的残差点数量参数赋值给实例变量 self.N_f
        self.N_f = N_f  # 方程点
        # 初始化实例变量 self.pts_bc 为 None，表示边界点坐标尚未定义
        self.pts_bc = None
    def loading_boundary_data(self):
        """ preparing boundary data """
        Nx = self.N_b
        Ny = self.N_b
        r_const = 10
        
        upper_x = np.random.uniform(self.x_min, self.x_max, Nx) #边界上是随机采样
        lower_x = np.random.uniform(self.x_min, self.x_max, Nx) #边界下采样
        left_y = np.random.uniform(self.y_min, self.y_max, Ny)  #边界左采样
        right_y = np.random.uniform(self.y_min, self.y_max, Ny) #边界右采样
        
        u_upper = np.sin(2*np.pi*upper_x)

        #  lower upper left right
        x_b = np.concatenate([lower_x,
                              upper_x,
                              self.x_min * np.ones_like(left_y),
                              self.x_max * np.ones_like(right_y)], 
                              axis=0).reshape([-1, 1])
        y_b = np.concatenate([self.y_min * np.ones_like(lower_x),
                              self.y_max * np.ones_like(upper_x),
                              left_y,
                              right_y],
                              axis=0).reshape([-1, 1])
        
        u_b = np.concatenate([np.zeros_like(lower_x),
                              u_upper,
                              np.zeros_like(left_y),
                              np.zeros_like(right_y)],
                              axis=0).reshape([-1, 1])

        v_b = np.zeros([u_b.shape[0]]).reshape([-1, 1])

        self.pts_bc = np.hstack((x_b,y_b))
      
        N_train_bcs = x_b.shape[0]
        print('-----------------------------')
        print('N_train_bcs: ' + str(N_train_bcs) )   #打印边界点的个数
        print('N_train_equ: ' + str(self.N_f) )      #打印内部采样点个数
        print('-----------------------------')     
        return x_b, y_b, u_b, v_b

    def loading_training_data(self):
        """ preparing training data """
        xye = LHSample(2, [[self.x_min, self.x_max], [self.y_min, self.y_max]], self.N_f)  #拉丁超立方采样
        if self.pts_bc is not None:
            xye_sorted, _ = sort_pts(xye, self.pts_bc)
        else:
            print("need to load boundary data first!")
            raise 
        x_train_f = xye_sorted[:, 0:1]
        y_train_f = xye_sorted[:, 1:2]
        return x_train_f, y_train_f

    '''
    def loading_evaluate_data(self, filename):
        """ preparing test data """
        data = scipy.io.loadmat(filename)
        x = data['X_ref']
        y = data['Y_ref']
        u = data['U_ref']
        v = data['V_ref']
        p = data['P_ref']
        x_star = x.reshape(-1,1)
        y_star = y.reshape(-1,1)
        u_star = u.reshape(-1,1)
        v_star = v.reshape(-1,1)
        p_star = p.reshape(-1,1)
        return x_star, y_star, u_star, v_star, p_star
    '''
    def loading_evaluate_data(self, filename):
        """ preparing test data """
        data = np.loadtxt(filename)
        x = data[:,0]
        y = data[:,1]
         
        x_star = x.reshape(-1,1)
        y_star = y.reshape(-1,1)

        return x_star, y_star

