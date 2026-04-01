import numpy as np
from scipy.integrate import solve_ivp

class VTOL_SF:
    # 气动系数
    paramsCL = np.array([-0.00275, 0.03228, 0.05795, -0.4521, -0.355, 1.252, 0.4043])
    paramsCD = np.array([0.01105, 0.001283, -0.2171, -0.02821, 1.045, 0.1516, 0.2304])
    paramsCM = np.array([0.0001976, -0.003862, -0.002853, 0.02184, 0.008538, 0.1649, 0.0219])
    Cn = 0.0
    Cl = 0.0
    Clp = 0.0
    Cmq = -0.5
    Cnr = 0.0
    # 基本参数
    g = np.array([0, 0, 9.8])
    m = 0.130 * (1 + 1/8*2)
    I = np.array([[1138.78/1e6, 7.66/1e6, 8.95/1e6],
                [7.66/1e6, 355.8/1e6, -4.97/1e6],
                [8.95/1e6, -4.97/1e6, 1478.23/1e6]])
    inv_I = np.array([[878.302730185509, 18.984106770982590, -5.38154444559513],
                [-18.9841067709826,	2811.11006898287, 9.56625477662147],
                [-5.38154444559513,	9.56625477662147, 676.549460577196]])

    S = 0.0604
    rho = 1.225
    cbar = 0.124
    b = 0.5
    l_armx = 0.0725
    l_army = 0.1
    # 推力系数
    CT = 2.618e-05 / (2*np.pi)**2

    def __init__(self, init_state=np.array([0,0,0, 0,0,np.pi/2, 0,0,0, 0.0,0.0,0.0], dtype=np.float32), dt=0.01):
        self.dt = dt
        # 状态初始化
        self.state = init_state  # [x,y,z, yaw,roll,pitch, vx,vy,vz, wx,wy,wz]

    # 采样还需考虑是否需要不同种子
    def rand_state_reset(self,
        pos_range=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)),      # (x,y,z) 范围
        att_range=((-30.0, 30), (-180.0, 180), (-30.0+90, 30+90)),  # (yaw,roll,pitch) 范围
        vel_range=((-1.0, 1), (-1.0, 1), (-1.0, 1)),          # (vx,vy,vz) 范围
        w_range=((-360.0, 360), (-180.0, 180), (-180.0, 180)),          # (wx,wy,wz) 范围
        seed=None
        ):
        if seed is not None:
            np.random.seed(seed)

        # 分别从均匀分布中采样
        x = np.random.uniform(*pos_range[0])
        y = np.random.uniform(*pos_range[1])
        z = np.random.uniform(*pos_range[2])

        yaw = np.random.uniform(*[v / 180 * np.pi for v in att_range[0]])
        roll = np.random.uniform(*[v / 180 * np.pi for v in att_range[1]])
        pitch = np.random.uniform(*[v / 180 * np.pi for v in att_range[2]])

        vx = np.random.uniform(*vel_range[0])
        vy = np.random.uniform(*vel_range[1])
        vz = np.random.uniform(*vel_range[2])

        wx = np.random.uniform(*[v / 180 * np.pi for v in w_range[0]])
        wy = np.random.uniform(*[v / 180 * np.pi for v in w_range[1]])
        wz = np.random.uniform(*[v / 180 * np.pi for v in w_range[2]])

        self.state = np.array([x, y, z, yaw, roll, pitch, vx, vy, vz, wx, wy, wz])
    
    def step(self, control):
        # state_dot = self.StateFcn_self_6DOF(1, self.state, control)
        # self.state = (self.state.astype(float) + state_dot * self.dt).astype(np.float32)
        # print(f"state_dot: {state_dot}")
        # success = 1

        sol = solve_ivp(
            fun=lambda t, y: self.StateFcn_self_6DOF(t, y, control),  # 固定ODE函数
            t_span=(0, self.dt),
            y0=self.state,
            method='RK23',  # 对应ode23，速度优先
            rtol=1e-2,      # 相对误差（放宽以提速）
            atol=1e-4,      # 绝对误差
            t_eval=[self.dt],    # 只输出最终时刻的状态（无需中间点）
            dense_output=False  # 关闭稠密输出（节省内存+时间）
        )
         # 更新状态（判断求解是否成功）
        success = sol.success
        if success:
            self.state = (sol.y[:, -1]).astype(np.float32)  # sol.y是二维数组，取最后一列（dt时刻状态）
        else:
            print(f"ODE求解失败！原因：{sol.message}")

        return self.state.copy(), success
    
    def StateFcn_self_6DOF(self, t, state, control):
        """
        六自由度无人机状态导数函数
        """

        state_64 = state.astype(float)
        control_64 = control.astype(float)

        # 提取状态
        yaw = state_64[3]
        roll = state_64[4]
        pitch = state_64[5]
        v = state_64[6:9]          # 地面速度
        # euler_dot = state_64[9:12]  # [yaw_dot, roll_dot, pitch_dot]
        # yaw_dot, roll_dot, pitch_dot = euler_dot
        wq = state_64[9:12]
        R_wq2deulerdot = np.array([[np.cos(pitch), 0, np.sin(pitch)],
            [np.sin(pitch)*np.tan(roll), 1, -np.cos(pitch)*np.tan(roll)],
            [-np.sin(pitch)/np.cos(roll), 0, np.cos(pitch)/np.cos(roll)]])
        temp_euler_dot = R_wq2deulerdot @ wq
        euler_dot = np.array([temp_euler_dot[2], temp_euler_dot[0], temp_euler_dot[1]])

        # 旋转矩阵 (机体 -> 地面)
        R_b2e = self.eul2rotm_zxy_symbolic(yaw, pitch, roll)

        # 机体速度
        vb = R_b2e.T @ v   # 等效于 R_e2b * v

        # 迎角
        alpha = np.arctan2(vb[2], vb[0])

        # 气动系数
        CL, _, _ = self.CX_dCX_ddCX_fit_single_seg(alpha, self.paramsCL)
        CD, _, _ = self.CX_dCX_ddCX_fit_single_seg(alpha, self.paramsCD)
        Cm, _, _ = self.CX_dCX_ddCX_fit_single_seg(alpha, self.paramsCM)

        # 控制量
        delta1 = control_64[0]
        delta2 = control_64[1]
        rpm1 = 2000 * control_64[2]
        rpm2 = 2000 * control_64[3]
        Th1 = self.CT * rpm1**2
        Th2 = self.CT * rpm2**2

        # 推力方向 (机体坐标系)
        R_de12b = np.array([[np.cos(delta1), 0, np.sin(delta1)],
                            [0,              1, 0],
                            [-np.sin(delta1),0, np.cos(delta1)]])
        R_de22b = np.array([[np.cos(delta2), 0, np.sin(delta2)],
                            [0,              1, 0],
                            [-np.sin(delta2),0, np.cos(delta2)]])
        e_de_x = np.array([1, 0, 0])
        F_Th1_b = Th1 * (R_de12b @ e_de_x)
        F_Th2_b = Th2 * (R_de22b @ e_de_x)
        F_Th = R_b2e @ (F_Th1_b + F_Th2_b)   # 推力在地面系

        # 气动力
        va_norm = np.linalg.norm(v)
        Q = 0.5 * self.rho * va_norm**2 * self.S
        fa_a = np.array([-Q*CD, 0, -Q*CL])    # 气流坐标系下的气动力
        R_a2b = np.array([[np.cos(alpha), 0, np.sin(alpha)],
                        [0,             1, 0],
                        [-np.sin(alpha),0, np.cos(alpha)]]).T  # 从气流到机体的旋转矩阵
        F_aero = R_b2e @ (R_a2b @ fa_a)      # 气动力在地面系

        # 重力
        F_G = self.m * self.g

        # 总外力 (地面系)
        F_mat = F_Th + F_aero + F_G

        # # 机体角速度 (从欧拉角速度转换)
        # R_eulerdot2wq = np.array([[np.cos(pitch), 0, -np.cos(roll)*np.sin(pitch)],
        #                         [0,              1,  np.sin(roll)],
        #                         [np.sin(pitch), 0,  np.cos(pitch)*np.cos(roll)]])
        # wq = R_eulerdot2wq @ euler_dot   # 机体角速度 [p; q; r]

        # 推力力矩 (机体坐标系)
        M_Th_x = -self.l_army * (F_Th1_b[2] - F_Th2_b[2])
        M_Th_y = -self.l_armx * (F_Th1_b[2] + F_Th2_b[2])
        M_Th_z =  self.l_army * (F_Th1_b[0] - F_Th2_b[0])
        M_Th = np.array([M_Th_x, M_Th_y, M_Th_z])

        # 气动力矩 (机体坐标系)
        M_a = Q * self.cbar * np.array([self.Cn, Cm, self.Cl])   # 注意 Cm 已计算

        # 阻尼导数 (简化)

        # 动压相关项
        M_q_b = np.diag([self.Clp, self.Cmq, self.Cnr]) @ (np.diag([self.b, self.cbar, self.b]) @ wq / (2*(va_norm + 1e-6))) * Q * self.cbar

        # 陀螺力矩
        M_I = -self.skew(wq) @ (self.I @ wq)

        M_aero_all = M_a + M_q_b + M_I
        M_mat = M_Th + M_aero_all

        # 加速度
        a_vec = F_mat / self.m   # 因为 m_mat 是对角阵，直接除法

        # 角加速度
        dw_vec = np.linalg.solve(self.I, M_mat)   # 求解 I * dw = M

        # # 从角加速度求欧拉角加速度
        # R_wq2deulerdot = np.array([[np.cos(pitch), 0, np.sin(pitch)],
        #                         [np.sin(pitch)*np.tan(roll), 1, -np.cos(pitch)*np.tan(roll)],
        #                         [-np.sin(pitch)/np.cos(roll), 0, np.cos(pitch)/np.cos(roll)]])
        # dR_eulerdot2wq = np.array([
        #     [-np.sin(pitch)*pitch_dot, 0, -(-np.sin(roll)*roll_dot*np.sin(pitch) + np.cos(roll)*np.cos(pitch)*pitch_dot)],
        #     [0, 1, np.cos(roll)*roll_dot],
        #     [np.cos(pitch)*pitch_dot, 0, (-np.sin(pitch)*pitch_dot*np.cos(roll) + np.cos(pitch)*(-np.sin(roll))*roll_dot)]
        # ])
        # deuler_dot = -R_wq2deulerdot @ dR_eulerdot2wq @ R_wq2deulerdot @ wq + R_wq2deulerdot @ dw_vec
        # dyaw_dot = deuler_dot[0]
        # droll_dot = deuler_dot[1]
        # dpitch_dot = deuler_dot[2]

        # 状态导数
        dydt = np.concatenate([v, euler_dot, a_vec, dw_vec])

        return dydt
    
    def CX_dCX_ddCX_fit_single_seg(self, alpha, coeffs):
        # 确保 coeffs 长度为 7
        c0, c1, c2, c3, c4, c5, c6 = coeffs
        # 计算函数值
        val = c0*alpha**6 + c1*alpha**5 + c2*alpha**4 + c3*alpha**3 + c4*alpha**2 + c5*alpha + c6
        # 一阶导数
        dval = 6*c0*alpha**5 + 5*c1*alpha**4 + 4*c2*alpha**3 + 3*c3*alpha**2 + 2*c4*alpha + c5
        # 二阶导数
        ddval = 30*c0*alpha**4 + 20*c1*alpha**3 + 12*c2*alpha**2 + 6*c3*alpha + 2*c4
        
        return val, dval, ddval
    
    def skew(self, v):
        """返回三维向量的反对称矩阵"""
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

    def eul2rotm_zxy_symbolic(self, yaw, pitch, roll):
        """
        欧拉角转旋转矩阵 (ZXY 顺序)
        输入: yaw(偏航), pitch(俯仰), roll(滚转)  (单位:弧度)
        输出: 从机体坐标系到地面坐标系的旋转矩阵 (3x3)
        """
        # 绕Z轴旋转 (yaw)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0,            0,           1]])
        # 绕X轴旋转 (roll)
        Rx = np.array([[1, 0,            0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll),  np.cos(roll)]])
        # 绕Y轴旋转 (pitch)
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0,             1, 0],
                    [-np.sin(pitch),0, np.cos(pitch)]])
        # ZXY 顺序: R = Rz @ Rx @ Ry
        R = Rz @ Rx @ Ry
        return R
    


if __name__ == "__main__":
    # vtol_test = VTOL_SF(dt=0.01,init_state=np.array([0,0,0, 0,0,np.pi/2, 2,2,2, 0.5,0.5,0.5]))
    vtol_test = VTOL_SF(dt=0.01)
    formatted_state = [f"{x:.2f}" for x in vtol_test.state]
    print(f"vtol_state: {formatted_state}")
    for i in range(1, 30):
        _, success = vtol_test.step(np.array([0.0, -0.0, 0.5, 0.5]))
        if success:
            formatted_state = [f"{x:.2f}" for x in vtol_test.state]
            print(f"{i}--vtol_state: {formatted_state}")
        else:
            break

    # vtol_test = VTOL_SF(dt=0.01)
    # state_dot = vtol_test.StateFcn_self_6DOF(1, vtol_test.state, np.array([0.0, 0.00, 0.0, 0.0]))
    