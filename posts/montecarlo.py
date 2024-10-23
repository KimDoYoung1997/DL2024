import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

class MonteCarlo:
    def __init__(self, n_rows, n_cols, goal_point, gamma=0.9, episodes=30, seed=42):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_states = n_rows * n_cols
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = gamma
        self.goal_point = goal_point
        self.v = np.zeros(self.n_states)
        self.returns = {state: [] for state in range(self.n_states)}
        self.frames = []
        self.iteration_counts = []
        self.episodes = episodes
        self.seed = seed
        
        # 시드 설정
        random.seed(self.seed)
        np.random.seed(self.seed)

    def transition_reward(self, state, action):
        row, col = divmod(state, self.n_cols)
        
        if action == 'up':
            next_row, next_col = max(row - 1, 0), col
        elif action == 'down':
            next_row, next_col = min(row + 1, self.n_rows - 1), col
        elif action == 'left':
            next_row, next_col = row, max(col - 1, 0)
        elif action == 'right':
            next_row, next_col = row, min(col + 1, self.n_cols - 1)
        
        next_state = next_row * self.n_cols + next_col
        reward = 1 if next_state == self.goal_point else -1
        return next_state, reward

    def generate_episode(self):
        episode = []
        state = random.randint(0, self.n_states - 1)
        
        while state != self.goal_point:
            action = random.choice(self.actions)
            next_state, reward = self.transition_reward(state, action)
            episode.append((state, action, reward))
            state = next_state
            
        return episode

    def monte_carlo(self):
        for episode_idx in range(self.episodes):
            episode = self.generate_episode()
            g = 0
            
            for state, _, reward in reversed(episode):
                g = reward + self.gamma * g
                if state not in [x[0] for x in episode[:episode.index((state, _, reward))]]:
                    self.returns[state].append(g)
                    self.v[state] = np.mean(self.returns[state])
            
            # 각 에피소드가 끝날 때마다 업데이트된 가치 함수를 저장
            self.frames.append(self.v.copy())
            self.iteration_counts.append(episode_idx + 1)
            # 현재 가치 함수 출력
            print(f"Step: {len(self.frames)}, State Value Function: {self.v}")

    def animate(self):
        fig, ax = plt.subplots(figsize=(6, 6))

        def update(frame_idx):
            ax.clear()
            v = self.frames[frame_idx]
            iteration = self.iteration_counts[frame_idx]
            ax.imshow(v.reshape(self.n_rows, self.n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=1)
            ax.set_title(f'Episode: {iteration}, goal_point: {self.goal_point}')

            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    state = i * self.n_cols + j
                    state_value = v[state]  # 에피소드별로 저장된 가치 함수의 상태 값 사용
                    ax.text(j, i, f'{state_value:.2f}', ha='center', va='center', color='black')

            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=1000, repeat=False)
        plt.show()

if __name__ == '__main__':
    goal_point = int(input('goal_point 의 index를 정수로 입력하세요 : '))
    mc = MonteCarlo(n_rows=4, n_cols=4, goal_point=goal_point)
    mc.monte_carlo()
    mc.animate()
