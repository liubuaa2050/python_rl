import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------- 策略网络（带标准差裁剪）----------
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64, log_std_min=-5, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        mean = self.net(x)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def get_action_logprob(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_logprob(self, state, action):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)

# ---------- 环境：简单一维控制，增加状态边界 ----------
class SimpleEnv:
    def __init__(self, goal=0.0, disturbance=0.0, max_state=10.0):
        self.goal = goal
        self.disturbance = disturbance
        self.max_state = max_state
        self.state = 0.0

    def reset(self):
        self.state = np.random.uniform(-1, 1)
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        # 更新状态
        self.state += action[0] + self.disturbance
        # 状态边界限制，并增加惩罚
        if abs(self.state) > self.max_state:
            self.state = np.clip(self.state, -self.max_state, self.max_state)
            reward = - (self.state - self.goal) ** 2 - 100.0
        else:
            reward = - (self.state - self.goal) ** 2
        # 奖励缩放，防止回报过大
        reward = reward * 0.01
        return np.array([self.state], dtype=np.float32), reward, False

# ---------- 采集轨迹 ----------
def collect_trajectories(env, policy, num_steps=20):
    states, actions, rewards = [], [], []
    state = env.reset()
    for _ in range(num_steps):
        s_t = torch.from_numpy(state).unsqueeze(0)
        a, _ = policy.get_action_logprob(s_t)
        a_np = a.detach().numpy().flatten()
        next_state, r, _ = env.step(a_np)
        states.append(state)
        actions.append(a)
        rewards.append(r)
        state = next_state

    # 计算折扣回报
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    # 标准化 returns（有助于稳定性）
    if returns.std() > 0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    else:
        returns = returns - returns.mean()

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.cat(actions, dim=0)
    return states, actions, returns

# ---------- FOMAML 训练 ----------
def fomaml_rl(meta_lr=1e-3, inner_lr=0.1, num_tasks=4, inner_steps=1,
              meta_iters=100, grad_clip=1.0):
    policy = Policy(state_dim=1, action_dim=1)
    meta_optimizer = optim.Adam(policy.parameters(), lr=meta_lr)

    for meta_iter in range(meta_iters):
        tasks = []
        for _ in range(num_tasks):
            goal = np.random.uniform(-2, 2)
            disturbance = np.random.uniform(-0.5, 0.5)
            tasks.append((goal, disturbance))

        meta_optimizer.zero_grad()
        task_losses = []

        for goal, dist in tasks:
            # 内循环
            fast_policy = Policy(state_dim=1, action_dim=1)
            fast_policy.load_state_dict(policy.state_dict())

            for _ in range(inner_steps):
                env = SimpleEnv(goal=goal, disturbance=dist)
                states, actions, returns = collect_trajectories(env, fast_policy, num_steps=20)
                log_probs = fast_policy.get_logprob(states, actions)
                loss = - (log_probs * returns).mean()
                grads = torch.autograd.grad(loss, fast_policy.parameters(), retain_graph=False)
                grads = [torch.clamp(g, -grad_clip, grad_clip) for g in grads]
                with torch.no_grad():
                    for param, grad in zip(fast_policy.parameters(), grads):
                        param -= inner_lr * grad

            # 外循环
            env_val = SimpleEnv(goal=goal, disturbance=dist)
            states_val, actions_val, returns_val = collect_trajectories(env_val, fast_policy, num_steps=20)
            log_probs_val = fast_policy.get_logprob(states_val, actions_val)
            meta_loss_task = - (log_probs_val * returns_val).mean()
            task_losses.append(meta_loss_task.item())

            task_grads = torch.autograd.grad(meta_loss_task, fast_policy.parameters(), retain_graph=False)

            for param, grad in zip(policy.parameters(), task_grads):
                if param.grad is None:
                    param.grad = grad.detach()
                else:
                    param.grad += grad.detach()

        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        meta_optimizer.step()

        if meta_iter % 10 == 0:
            avg_loss = np.mean(task_losses)
            print(f"Iter {meta_iter}, average meta loss: {avg_loss:.4f}")

    return policy

# ---------- 测试适应能力 ----------
def test_adaptation(policy, test_goal, test_disturb, inner_steps=1):
    fast_policy = Policy(state_dim=1, action_dim=1)
    fast_policy.load_state_dict(policy.state_dict())
    for _ in range(inner_steps):
        env = SimpleEnv(goal=test_goal, disturbance=test_disturb)
        states, actions, returns = collect_trajectories(env, fast_policy, 20)
        log_probs = fast_policy.get_logprob(states, actions)
        loss = - (log_probs * returns).mean()
        grads = torch.autograd.grad(loss, fast_policy.parameters())
        grads = [torch.clamp(g, -1, 1) for g in grads]
        with torch.no_grad():
            for p, g in zip(fast_policy.parameters(), grads):
                p -= 0.1 * g

    env_eval = SimpleEnv(goal=test_goal, disturbance=test_disturb)
    total_r = 0
    for _ in range(5):
        state = env_eval.reset()
        ep_r = 0
        for t in range(50):
            s_t = torch.from_numpy(state).unsqueeze(0)
            a, _ = fast_policy.get_action_logprob(s_t)
            state, r, _ = env_eval.step(a.detach().numpy().flatten())
            ep_r += r
        total_r += ep_r
    return total_r / 5

if __name__ == "__main__":
    trained_policy = fomaml_rl(meta_iters=200)
    avg_reward = test_adaptation(trained_policy, test_goal=1.5, test_disturb=0.3)
    print(f"Test adaptation reward: {avg_reward:.2f}")