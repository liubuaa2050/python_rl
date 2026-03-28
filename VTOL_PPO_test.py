from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
import os
import pandas as pd
from datetime import datetime

from utils.VTOL_env import VTOLEnv
from utils.Visualizer import Visualizer, VisualCallback

class Train_self:
    def __init__(self, total_timesteps=1e4):
        self.gpu_check()
        self.model_env_init(total_timesteps)

    def model_env_init(self, total_timesteps):
        self.total_timesteps = total_timesteps
        # 创建日志目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f"training_logs/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"训练日志将保存到: {self.log_dir}")

        # 环境构建+校验
        self.env = VTOLEnv()
        check_env(self.env, warn=True)

        # 创建可视化器
        self.visualizer = Visualizer(self.log_dir)

        # 创建评估环境
        eval_env = VTOLEnv()
        eval_env = Monitor(eval_env, f"{self.log_dir}/eval")

        # 创建回调
        self.eval_callback = VisualCallback(
            self.visualizer,
            eval_env,
            best_model_save_path=f"{self.log_dir}/best_model",
            log_path=self.log_dir,
            eval_freq=2000,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )

        # 网络结构
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[32, 64, 64, 32], vf=[32, 64, 64, 32]), #1.8.0之后不支持共享网络层
        )

        # 创建模型
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"{self.log_dir}/tensorboard",
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device='cpu'
        )
        print(self.model.policy)

    def train_with_monitor(self):
        print("\n开始训练...")
        print("=" * 50)

        # 训练
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=self.eval_callback,
            progress_bar=True
        )

        # 保存最终模型
        self.model.save(f"{self.log_dir}/final_model")
        
        # 保存最终可视化
        self.visualizer.save_final_plot(f"{self.log_dir}/training_summary.png")
        
        print(f"\n训练完成！所有文件保存在: {self.log_dir}")
        print(f"使用以下命令查看TensorBoard: tensorboard --logdir {self.log_dir}")


    def test_nn_model(self, step_num=1000):
        # 模型测试
        state = self.env.reset()
        for _ in range(step_num):
            action, _ = self.model.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        print("\n****测试完成****\n")

    def gpu_check(self):
        # 检测可用的GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")


if __name__ == "__main__":

    train_ppo = Train_self(total_timesteps=1e5)

    # train_ppo.train_with_monitor()

    