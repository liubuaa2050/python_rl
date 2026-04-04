from stable_baselines3 import PPO
import torch as th
import numpy as np

path_dir = "./training_logs/20260403_163009/"
model = PPO.load(path_dir+"best_model/best_model.zip", device='cpu')
model.policy.eval()

model_nn = th.jit.load(path_dir+'best_ppo_actor.pt')
model_nn.eval()

x_current = np.array([-2.597756567584874,1.778631712563411,-1.757582240504165,-0.170232779536826,2.650931252360165,1.255911798677078,0.327027699193169,-0.512050743246596,0.641988885474704,1.248809097603922,0.195341409488036,-0.598799193619527], dtype=np.float32)
x_current_th = th.tensor(x_current).unsqueeze(0)

print(model.policy.predict(x_current_th, deterministic=True))

print(model_nn.forward(x_current_th))
