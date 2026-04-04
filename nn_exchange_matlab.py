import torch as th
import os
from stable_baselines3 import PPO

class ActorWrapper(th.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.policy = policy

    def forward(self, obs):
        # 特征提取（对于 1D 输入，FlattenExtractor 只是展平，此处 obs 已经是 batch x 12）
        features = self.features_extractor(obs)
        # 共享 MLP 的 actor 分支输出 32 维特征
        latent_pi = self.mlp_extractor.forward_actor(features)
        # 动作网络输出 4 维（连续动作的均值或离散 logits）
        action = self.action_net(latent_pi)
        # action = self.policy.forward(obs, deterministic=True)
        return action
    
path_dir = "./training_logs/20260403_163009"
model = PPO.load(path_dir+"/best_model/best_model.zip", device='cpu')
model.policy.eval()

action_net = ActorWrapper(model.policy)
action_net.eval()
# print("=== action_net 结构 ===")
# print(action_net)

example_input = th.randn(1, 12)
# print(f"example_input: {example_input}")

with th.no_grad():
    traced_model = th.jit.trace(action_net, example_input)

export_path = os.path.join(path_dir, "best_ppo_actor.pt")
traced_model.save(export_path)
print("模型已成功导出")