import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

class PolicyImprovement:
    def __init__(self, n_rows, n_cols, goal_point, gamma=0.9, theta=1e-6):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_states = n_rows * n_cols
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = gamma
        self.theta = theta
        self.v = np.zeros(self.n_states)  # 상태 가치 함수 초기화
        self.policy = np.ones((self.n_states, len(self.actions))) / len(self.actions)  # 랜덤 정책으로 초기화 (각 행동에 대해 동일한 확률)
        self.goal_point = goal_point
        self.frames = []
        self.iteration_counts = []
        self.convergence_flags = []

    def transition_reward(self, state, action):
        # 상태를 그리드 좌표 (row, col)로 변환
        row, col = divmod(state, self.n_cols)
        # 각 행동에 따라 다음 상태 계산
        if action == 'up':
            next_row, next_col = max(row - 1, 0), col
        elif action == 'down':
            next_row, next_col = min(row + 1, self.n_rows - 1), col
        elif action == 'left':
            next_row, next_col = row, max(col - 1, 0)
        elif action == 'right':
            next_row, next_col = row, min(col + 1, self.n_cols - 1)

        # 다음 상태 계산 (좌표를 상태 번호로 변환)
        next_state = next_row * self.n_cols + next_col
        reward = 10 if next_state == self.goal_point - 1 else -1  # 목표 상태 도착 시 보상은 10, 나머지는 -1
        return next_state, reward

    def policy_evaluation(self):
        k = 0  # iteration 카운트
        while True:
            delta = 0
            v_new_states = np.zeros(self.n_states)
            # 모든 상태에 대해 가치 업데이트
            for state in range(self.n_states):
                v_new = 0
                # 현재 상태에서 가능한 모든 행동에 대해 기대 가치 계산
                for action_idx, action_prob in enumerate(self.policy[state]):
                    next_state, reward = self.transition_reward(state, self.actions[action_idx])
                    v_new += action_prob * (reward + self.gamma * self.v[next_state])
                delta = max(delta, abs(v_new - self.v[state]))
                v_new_states[state] = v_new
            self.v[:] = v_new_states[:]

            # 애니메이션 프레임 저장 (모든 상태가 업데이트된 후 저장)
            self.frames.append(self.v.copy())
            self.iteration_counts.append(k)
            self.convergence_flags.append(delta < self.theta)

            # 변화량이 충분히 작으면 (theta 이하이면) 종료
            if delta < self.theta:
                break
            k += 1  # iteration 증가

    def animate(self):
        fig, ax = plt.subplots(figsize=(6, 6))

        def update(frame_idx):
            ax.clear()
            v = self.frames[frame_idx]
            iteration = self.iteration_counts[frame_idx]
            converged = self.convergence_flags[frame_idx]
            ax.imshow(v.reshape(self.n_rows, self.n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=10)
            ax.set_title(f'Iteration: {iteration} - Converged: {converged}')

            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    state_value = v[i * self.n_cols + j]
                    ax.text(j, i, f'{state_value:.2f}', ha='center', va='center', color='black')

            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=100, repeat=False)
        plt.show()

if __name__ == '__main__':
    goal_point = int(input('goal_point 의 index를 정수로 입력하세요 : '))
    pe = PolicyImprovement(n_rows=7, n_cols=7, goal_point=goal_point)
    pe.policy_evaluation()
    pe.animate()