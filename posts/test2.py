import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import logging

# Logging 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 4x4 그리드 환경 정의
n_rows, n_cols = 4, 4
n_states = n_rows * n_cols

# 행동 정의 (상, 하, 좌, 우)
actions = ['up', 'down', 'left', 'right']

# 상태 전이 확률 및 보상
def transition_reward(state, action):
    row, col = divmod(state, n_cols)  # 상태를 그리드 좌표로 변환
    if action == 'up':
        next_row = max(row - 1, 0)
        next_col = col
    elif action == 'down':
        next_row = min(row + 1, n_rows - 1)
        next_col = col
    elif action == 'left':
        next_row = row
        next_col = max(col - 1, 0)
    elif action == 'right':
        next_row = row
        next_col = min(col + 1, n_cols - 1)

    next_state = next_row * n_cols + next_col
    reward = 1 if next_state == n_states - 1 else -1  # 목표 상태 도착 시 보상은 1, 나머지는 -1
    return next_state, reward

# 할인율
gamma = 0.9

# 상태 가치 함수 초기화
v = np.zeros(n_states)

# 애니메이션을 위한 데이터 저장
frames = []
iteration_counts = []
convergence_flags = []

# 가치 반복 함수
def value_iteration(gamma=0.9, theta=1e-6):
    k = 0  # iteration 카운트
    while True:
        delta = 0
        logging.info(f"Iteration {k} start")
        for state in range(n_states):
            # 현재 상태의 기존 가치
            old_value = v[state]
            v_new = float('-inf')
            
            # 현재 상태에서 가능한 모든 행동에 대해 최대 가치 계산
            for action in actions:
                next_state, reward = transition_reward(state, action)
                # 벨만 최적 방정식: v_new = max(reward + gamma * v[next_state])
                v_new = max(v_new, reward + gamma * v[next_state])
                
                logging.info(f'State {state}, Action {action} -> Next State {next_state}, Reward {reward}, Next Value {v[next_state]:.2f}')
            
            # 상태 가치 함수 업데이트 (벨만 최적 방정식 사용)
            v[state] = v_new
            
            # 변화량 계산 (현재 상태 가치와 업데이트된 상태 가치의 차이)
            delta = max(delta, abs(v_new - old_value))
            
            logging.info(f'State {state} - Old Value: {old_value:.2f}, New Value: {v_new:.2f}')
            
            # 애니메이션 프레임 저장 (각 상태 업데이트 후 저장하여 더 세밀한 시각화 제공)
            frames.append(v.copy())
            iteration_counts.append(k)
            convergence_flags.append(delta < theta)

        # 가치 함수가 충분히 수렴하면 종료
        if delta < theta:
            logging.info('가치 함수가 수렴하였습니다.')
            break
        
        k += 1

# 가치 반복 실행
value_iteration(gamma)

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(6, 6))

def update(frame_idx):
    ax.clear()
    v = frames[frame_idx]
    iteration = iteration_counts[frame_idx]
    converged = convergence_flags[frame_idx]
    ax.imshow(v.reshape(n_rows, n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=1)
    ax.set_title(f'Iteration: {iteration} - Converged: {converged}')
    
    for i in range(n_rows):
        for j in range(n_cols):
            state_value = v[i * n_cols + j]
            ax.text(j, i, f'{state_value:.2f}', ha='center', va='center', color='black')

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 애니메이션 실행
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, repeat=False)
plt.show()