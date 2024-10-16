import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation  # 전체 animation 모듈을 가져옵니다.

# 4x4 그리드 환경 정의
n_rows, n_cols = 4, 4
n_states = n_rows * n_cols

# 행동 정의 (상, 하, 좌, 우)
actions = ['up', 'down', 'left', 'right']

# 정책 정의 (랜덤 정책으로 설정)
policy = np.ones((n_states, len(actions))) / len(actions)

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
    reward = -1  # 모든 이동의 보상은 -1로 설정
    return next_state, reward

# 할인율
gamma = 0.9

# 상태 가치 함수 초기화
v = np.zeros(n_states)

# 애니메이션을 위한 데이터 저장
frames = []

# 정책 평가 함수
def policy_evaluation(policy, gamma=0.9, theta=1e-6):
    k = 0  # iteration 카운트
    while True:
        delta = 0
        for state in range(n_states):
            v_new = 0
            # 현재 상태에서 가능한 모든 행동에 대해 기대 가치 계산
            for action_idx, action_prob in enumerate(policy[state]):
                next_state, reward = transition_reward(state, actions[action_idx])
                v_new += action_prob * (reward + gamma * v[next_state])
            
            delta = max(delta, abs(v_new - v[state]))
            v[state] = v_new  # 상태 가치 함수 업데이트

            # 애니메이션 프레임 저장
            frames.append((v.copy(), state, actions[action_idx]))

        if delta < theta:
            break
        k += 1

# 정책 평가 실행
policy_evaluation(policy, gamma)

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()
    v, state, action = frame
    ax.imshow(v.reshape(n_rows, n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=0)
    ax.set_title(f'State Value Function\nState: {state}, Action: {action}')
    
    for i in range(n_rows):
        for j in range(n_cols):
            state_value = v[i * n_cols + j]
            ax.text(j, i, f'{state_value:.3f}', ha='center', va='center', color='black')

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 애니메이션 실행
ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, repeat=False)
plt.show()
