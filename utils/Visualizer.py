from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """训练可视化器"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        plt.ion()
        
    def update_plots(self):
        """更新所有图表"""
        # 加载训练数据
        results = load_results(self.log_dir)
        
        if len(results) == 0:
            return
            
        # 清除旧内容
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        # 使用 Episode 序号作为横坐标
        episodes = np.arange(len(results))
        
        # 1. 奖励曲线
        self.ax1.plot(episodes, results['r'], 'b-', alpha=0.3, label='Episode Reward')
        if len(results) >= 10:
            moving_avg = results['r'].rolling(window=10).mean()
            self.ax1.plot(episodes, moving_avg, 'r-', label='Moving Avg (10)')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.set_title('Training Rewards')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # 2. 回合长度
        self.ax2.plot(episodes, results['l'], 'g-', alpha=0.5)
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Length')
        self.ax2.set_title('Episode Lengths')
        self.ax2.grid(True)
        
        # 3. 累计奖励直方图
        self.ax3.hist(results['r'], bins=20, color='skyblue', edgecolor='black')
        self.ax3.set_xlabel('Reward')
        self.ax3.set_ylabel('Frequency')
        self.ax3.set_title('Reward Distribution')
        self.ax3.grid(True, alpha=0.3)
        
        # 4. 训练进度统计
        stats_text = f"""
        train imformation:
        =============
        episodes: {len(results)}
        reward_mean: {results['r'].mean():.2f}
        reward_best: {results['r'].max():.2f}
        reward_worst: {results['r'].min():.2f}
        episode_step_mean: {results['l'].mean():.1f}
        episode_step_mean_in10: {results['r'].tail(10).mean():.2f}
        """
        self.ax4.text(0.1, 0.5, stats_text, transform=self.ax4.transAxes, 
                     fontsize=12, verticalalignment='center',
                     fontfamily='monospace')
        self.ax4.axis('off')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
    def save_final_plot(self, filename):
        """保存最终图表"""
        self.update_plots()
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {filename}")

class VisualCallback(EvalCallback):
    """带有可视化的评估回调"""
    def __init__(self, visualizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visualizer = visualizer
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        # 定期更新可视化
        if self.n_calls % 1000 == 0:
            self.visualizer.update_plots()
        return result