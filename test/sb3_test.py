'''
Author: INFINITY 2040568761@qq.com
Date: 2026-03-12 16:39:54
LastEditors: INFINITY 2040568761@qq.com
LastEditTime: 2026-03-12 16:39:59
FilePath: \python_rl\sb3_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import gymnasium as gym
from stable_baselines3 import PPO
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0)
print("🌟 环境初始化成功！")
print("🔥 可开始训练强化学习模型！")