import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')

class ValueIteration:
    def __init__(self, n_rows, n_cols, goal_point, gamma=0.9, theta=1e-6):
        self.n_rows = n_rows  # 격자의 행 수
        self.n_cols = n_cols  # 격자의 열 수
        self.n_states = n_rows * n_cols  # 상태의 총 수 (격자의 모든 칸)
        self.actions = ['up', 'down', 'left', 'right']  # 가능한 행동 리스트
        self.gamma = gamma  # 할인 계수 (감가율)
        self.theta = theta  # 가치 함수 변화량의 기준 (수렴 판정 기준)
        self.v = np.zeros(self.n_states)  # 초기 가치 함수 설정 (모든 상태의 가치를 0으로 초기화)
        self.goal_point = goal_point  # 목표 상태 설정
        self.frames = []  # 애니메이션을 위한 가치 함수 저장 리스트
        self.iteration_counts = []  # 각 반복의 횟수를 저장

    def transition_reward(self, state, action):
        row, col = divmod(state, self.n_cols)  # 현재 상태의 행과 열 계산
        
        # 행동에 따른 다음 상태 계산
        if action == 'up':
            next_row, next_col = max(row - 1, 0), col
        elif action == 'down':
            next_row, next_col = min(row + 1, self.n_rows - 1), col
        elif action == 'left':
            next_row, next_col = row, max(col - 1, 0)
        elif action == 'right':
            next_row, next_col = row, min(col + 1, self.n_cols - 1)
        
        next_state = next_row * self.n_cols + next_col  # 다음 상태의 인덱스 계산
        reward = 1 if next_state == self.goal_point else -1  # 목표 상태에 도달하면 +1, 그렇지 않으면 -1의 보상
        return next_state, reward

    def value_iteration(self):
        print("value_iteration")
        while True:
            delta = 0  # 가치 함수 변화량 초기화
            v_new_states = np.zeros(self.n_states)  # 새로운 가치 함수를 저장할 배열 초기화
            
            # 각 상태에 대해 가치 함수 갱신
            for state in range(self.n_states):
                action_values = []
                
                # 가능한 모든 행동에 대해 행동 가치를 계산
                for action in self.actions:
                    next_state, reward = self.transition_reward(state, action)
                    action_value = reward + self.gamma * self.v[next_state]
                    action_values.append(action_value)
                
                # 현재 상태의 새로운 가치는 가능한 모든 행동 가치 중 최대값
                v_new_states[state] = max(action_values)
                
                # 가치 함수 변화량 추적
                delta = max(delta, abs(v_new_states[state] - self.v[state]))
            
            # 가치 함수 갱신
            self.v[:] = v_new_states[:]
            
            # 애니메이션 프레임 저장
            self.frames.append(self.v.copy())
            self.iteration_counts.append(len(self.iteration_counts))

            # 가치 함수의 변화량이 기준치보다 작아지면 수렴했다고 판단하고 반복 종료
            if delta < self.theta:
                break

    def animate(self):
        fig, ax = plt.subplots(figsize=(6, 6))

        def update(frame_idx):
            ax.clear()
            v = self.frames[frame_idx]
            iteration = self.iteration_counts[frame_idx]
            ax.imshow(v.reshape(self.n_rows, self.n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=1)
            ax.set_title(f'Iteration: {iteration}, Destination: {self.goal_point}')

            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    state = i * self.n_cols + j
                    state_value = v[state]
                    ax.text(j, i, f'{state_value:.2f}', ha='center', va='center', color='black')

            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=300, repeat=False)
        plt.show()

if __name__ == '__main__':
    goal_point = int(input('goal_point 의 index를 정수로 입력하세요 : '))
    vi = ValueIteration(n_rows=7, n_cols=7, goal_point=goal_point)
    vi.value_iteration()
    vi.animate()
