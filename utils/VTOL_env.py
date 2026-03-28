import gymnasium as gym
from gymnasium import spaces
import numpy as np
# from VTOL_dynamic import VTOL_SF

if __name__ == "__main__":
    # 直接运行时使用绝对导入
    from VTOL_dynamic import VTOL_SF
else:
    # 作为模块导入时使用相对导入
    from .VTOL_dynamic import VTOL_SF

class VTOLEnv(gym.Env):
    x_error_Thold = 1e4
    y_error_Thold = 1e4
    z_error_Thold = 1e4
    vx_Thold = 1e2
    vy_Thold = 1e2
    vz_Thold = 1e2
    yaw_Thold = 80 * np.pi/180
    pitch_Thold = 80 * np.pi/180
    roll_Thold = 180 * np.pi/180
    wx_Thold = 1e3 * np.pi/180
    wy_Thold = 1e3 * np.pi/180
    wz_Thold = 1e3 * np.pi/180

    def __init__(self, max_step=1000):
        super().__init__()
        # 定义动作空间和观测空间（例如）
        self.action_space = spaces.Box(low=np.array([-np.pi, -np.pi, 0, 0],dtype=np.float32), high=np.array([np.pi, np.pi, 1, 1],dtype=np.float32), shape=(4,),)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,))

        # 后续可以考虑进行动作归一化
        
        # 初始化你的无人机模型
        self.VTOL = VTOL_SF(dt=0.01)  # 或 PyBullet 接口
        self.current_step = 0
        self.max_step = max_step
        
    def reset(self, seed=None, options=None):
        # 重置无人机状态
        self.current_step = 0
        self.VTOL.rand_state_reset()
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        # 执行一步仿真
        self.current_step += 1
        # action 可能需要缩放为实际物理量（如转速）
        control = action
        self.VTOL.step(control)
        obs = self._get_obs()
        terminated = self._check_terminated(obs)
        if terminated:
            reward = 0
        else:
            reward = self._compute_reward(obs, action)
        truncated = self.current_step >= self.max_step  # 可根据需要设置（如最大步数）=
        info = {}
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        # 返回观测向量
        return self.VTOL.state.astype(np.float32)
    
    def _compute_reward(self, obs, action):
        # 自定义奖励函数
        pos = obs[:3]
        eul = obs[3:6]
        vel = obs[6:9]
        wq = obs[9:]
        Cp = 1e-3
        Cq = 10
        Cv = 1e-3
        Cw = 1e-1
        Ca = 1
        SumReward = - Cp*np.linalg.norm(pos-np.array([0,0,0])) - Cq*np.linalg.norm(eul[[0,2]]-np.array([0,np.pi/2])) - Cv*np.linalg.norm(vel) \
            - Cw*np.linalg.norm(wq) - Ca*(np.linalg.norm(action-np.array([0,0,0.51,0.51])))
        
        reward = np.exp(SumReward)
        return reward
    
    def _check_terminated(self, obs):
        # 判断是否坠毁或超出边界
        pos = obs[:3]
        eul = obs[3:6]
        vel = obs[6:9]
        wq = obs[9:]
        IsStateNorm = abs(pos[0]) > self.x_error_Thold or abs(pos[1]) > self.y_error_Thold or abs(pos[2]) > self.z_error_Thold or \
            abs(vel[0]) > self.vx_Thold or abs(vel[1]) > self.vy_Thold or abs(vel[2]) > self.vz_Thold or \
            abs(eul[0]) > self.yaw_Thold or abs(eul[2]-np.pi/2) > self.pitch_Thold or \
            abs(wq[0]) > self.wx_Thold or abs(wq[1]) > self.wy_Thold or abs(wq[2]) > self.wz_Thold

        IsStateNAN = np.isnan(obs).any()

        IsDone = IsStateNAN | IsStateNorm

        return bool(IsDone)
    
    def close(self):
        # 清理资源
        pass

if __name__ == "__main__":
    env = VTOLEnv()
    print(f"初始状态: {env._get_obs()}")
    env.reset()
    print(f"随机状态: {env._get_obs()}")
    env.step(np.array([0.0, 0.0, 0.5, 0.5]))
    print(f"步进状态: {env._get_obs()}")
    env.close()