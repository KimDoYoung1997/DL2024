import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation  # 전체 animation 모듈을 가져옵니다.

# 4x4 그리드 환경 정의
n_rows, n_cols = 4, 4
n_states = n_rows * n_cols
goal_state = 15  # 목표 상태

# 행동 정의 (상, 하, 좌, 우)
actions = ['up', 'down', 'left', 'right']

# 정책 정의 (랜덤 정책으로 설정)
policy = np.ones((n_states, len(actions))) / len(actions)

# 상태 전이 및 보상 함수 정의
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
    if next_state == goal_state:
        reward = 10  # 목표 상태에 도달하면 보상 +10
    return next_state, reward

# 할인율 및 수렴 기준
gamma = 0.9
theta = 1e-6

# 상태 가치 함수 초기화
v = np.zeros(n_states)

# 애니메이션을 위한 데이터 저장
frames = []

# Value Iteration 함수
def value_iteration(gamma=0.9, theta=1e-6):
    k = 0  # iteration 카운트
    while True:
        delta = 0
        new_v = v.copy()
        for state in range(n_states):
            if state == goal_state:  # 목표 상태는 가치가 0으로 고정됨
                continue
            action_values = []
            for action in actions:
                next_state, reward = transition_reward(state, action)
                action_value = reward + gamma * v[next_state]
                action_values.append(action_value)
            new_v[state] = max(action_values)  # 최적의 행동 선택 (max)
            delta = max(delta, abs(new_v[state] - v[state]))

            # 애니메이션 프레임 저장
            frames.append((new_v.copy(), state, actions[np.argmax(action_values)]))

        v[:] = new_v  # 상태 가치 함수 업데이트

        if delta < theta:  # 수렴 조건
            break
        k += 1

# Value Iteration 실행
value_iteration(gamma)

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()
    v, state, action = frame
    ax.imshow(v.reshape(n_rows, n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=10)
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