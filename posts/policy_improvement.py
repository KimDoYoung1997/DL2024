import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')

class PolicyIteration:
    def __init__(self, n_rows, n_cols, goal_point, gamma=0.9, theta=1e-6):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_states = n_rows * n_cols
        self.actions = ['up', 'down', 'left', 'right']
        self.gamma = gamma
        self.theta = theta
        self.v = np.zeros(self.n_states)
        self.policy = np.ones((self.n_states, len(self.actions))) / len(self.actions)  # 랜덤 정책으로 초기화 (각 행동에 대해 동일한 확률)  # 랜덤 정책으로 초기화
        self.goal_point = goal_point
        self.frames = []
        self.iteration_counts = []
        self.convergence_flags = []
        self.policies = [self.policy.copy()]  # 초기 정책을 저장

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

    def policy_evaluation(self):
        print("policy_evaluation")
        while True:
            delta = 0
            v_new_states = np.zeros(self.n_states)
            for state in range(self.n_states):
                v_new = 0
                for action_idx, action_prob in enumerate(self.policy[state]):  # 각 상태에서 모든 행동에 대한 기대 가치 계산
                    next_state, reward = self.transition_reward(state, self.actions[action_idx])
                    v_new += action_prob * (reward + self.gamma * self.v[next_state])
                delta = max(delta, abs(v_new - self.v[state]))  # 가치 함수의 변화량 추적
                v_new_states[state] = v_new
            self.v[:] = v_new_states[:]

            self.frames.append(self.v.copy())
            self.iteration_counts.append(len(self.iteration_counts))
            self.convergence_flags.append(delta < self.theta)

            if delta < self.theta:  # 가치 함수의 변화량이 기준치보다 작아지면 평가 종료
                break

    def policy_improvement(self):
        print("policy_improvement")

        # 새로운 개선된 정책을 저장하기 위한 배열을 초기화합니다.
        # 크기는 (상태의 수, 가능한 행동의 수)로 각 상태에서 각 행동을 선택할 확률을 나타냅니다.
        new_policy = np.zeros((self.n_states, len(self.actions)))

        # 각 상태(state)에 대해 정책을 개선합니다.
        for state in range(self.n_states):
            # 현재 상태에서 가능한 모든 행동의 가치를 계산하기 위해 리스트를 초기화합니다.
            action_values = []

            # 현재 상태에서 가능한 모든 행동(action)에 대해 반복합니다.
            for action in self.actions:
                # transition_reward 함수는 현재 상태와 행동을 받아서 다음 상태와 보상을 반환합니다.
                next_state, reward = self.transition_reward(state, action)

                # 행동 가치(action_value)를 계산합니다.
                # 행동 가치 = 현재 행동으로 얻는 보상 + 감가율(gamma) * 다음 상태의 가치
                # 다음 상태의 가치는 self.v[next_state]로 가져옵니다.
                action_value = reward + self.gamma * self.v[next_state]

                # 계산한 행동 가치를 action_values 리스트에 추가합니다.
                action_values.append(action_value)

            # 모든 가능한 행동 가치 중에서 가장 높은 가치를 갖는 행동의 인덱스를 찾습니다.
            # np.argmax(action_values)는 action_values 리스트에서 가장 큰 값을 가지는 인덱스를 반환합니다.
            best_action_idx = np.argmax(action_values)

            # 개선된 정책에서 현재 상태에서 가장 좋은 행동의 확률을 1로 설정합니다.
            # (즉, 가장 좋은 행동만을 선택하는 결정적 정책을 만듭니다.)
            new_policy[state][best_action_idx] = 1.0

        # 개선된 정책을 policies 리스트에 저장하여 정책 변화 과정을 추적할 수 있게 합니다.
        self.policies.append(new_policy)

        # 최종적으로 개선된 정책을 반환합니다.
        return new_policy
    
    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            new_policy = self.policy_improvement()
            if np.allclose(new_policy, self.policy, atol=1e-3):  # 개선된 정책이 이전 정책과 거의 동일하면 반복 종료
                break
            self.policy = new_policy

    def animate(self):
        fig, ax = plt.subplots(figsize=(6, 6))

        def update(frame_idx):
            ax.clear()
            v = self.frames[frame_idx]
            iteration = self.iteration_counts[frame_idx]
            converged = self.convergence_flags[frame_idx]
            ax.imshow(v.reshape(self.n_rows, self.n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=1)
            ax.set_title(f'Iteration: {iteration} - Converged: {converged}, Destination{self.goal_point}')

            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    state = i * self.n_cols + j
                    state_value = v[state]
                    ax.text(j, i, f'{state_value:.2f}', ha='center', va='center', color='black')
                    action_idx = np.argmax(self.policies[min(frame_idx, len(self.policies) - 1)][state])
                    action_str = ['↑', '↓', '←', '→'][action_idx]
                    ax.text(j, i + 0.3, f'{action_str}', ha='center', va='center', color='blue')

            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=300, repeat=False)
        plt.show()

if __name__ == '__main__':
    goal_point = int(input('goal_point 의 index를 정수로 입력하세요 : '))
    pi = PolicyIteration(n_rows=7, n_cols=7, goal_point=goal_point)
    pi.policy_iteration()
    pi.animate()