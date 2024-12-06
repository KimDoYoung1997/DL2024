# 정책 평가는 주어진 정책이 각 상태에서 얼마나 좋은지(얼마나 큰 보상을 기대할 수 있는지)를 계산하기 위해 상태 가치 함수 V(s)를 반복적으로 업데이트합니다. 
# 이는 정책 개선(policy improvement) 단계의 기초가 됩니다. 정책 평가는 현재의 정책을 따랐을 때 각 상태에서 얻을 수 있는 예상 보상의 총합을 구하지만, 정책 자체를 수정하거나 개선하지는 않습니다.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time

# 4x4 그리드 환경 정의 (총 49개의 상태를 가지는 환경)
n_rows, n_cols = 4, 4
n_states = n_rows * n_cols

# 행동 정의 (상, 하, 좌, 우)
actions = ['up', 'down', 'left', 'right']

# 할인율 (감가율)
gamma = 0.9

# 상태 가치 함수 초기화 (모든 상태의 가치를 0으로 초기화)
v = np.zeros(n_states)
v_new_states = np.zeros(n_states)
# 정책 정의 (랜덤 정책으로 설정) -> 모든 상태에서 네 방향으로 같은 확률로 이동
# 각 상태에서 모든 방향으로 동일한 확률로 이동하도록 설정됨 (즉, 각 행동의 확률이 0.25임)
# 이는 특정 방향성 없이 임의로 움직이는 정책을 의미하며, 결과적으로 가치 함수가 특정 목표로 수렴하지 못하고 모든 상태의 가치가 비슷하게 분포될 수 있음
policy = np.ones((n_states, len(actions))) / len(actions)
global goal_points 
goal_points = int(input('goal_point 의 index를 정수로 입력하세요 : '))

# 상태 전이 확률 및 보상
# 현재 상태와 행동에 따라 다음 상태와 보상을 계산하는 함수
def transition_reward(state, action):
    # 상태를 그리드 좌표 (row, col)로 변환
    global goal_points
    row, col = divmod(state, n_cols)
    print("---------------------- transition_reward 함수 실행 ----------------------")
    print(f"row : {row}, col : {col}")

    # 각 행동에 따라 다음 상태 계산
    if action == 'up':
        print(f"case 1 의 row-1={row-1}")
        next_row = max(row - 1, 0)  # 위로 이동할 때, 그리드 범위를 벗어나지 않도록 제한
        next_col = col
        print(f"next_row : {next_row} , next_col : {next_col}")

    elif action == 'down':
        print(f"case 2 의 row+1={row+1} , n_rows-1={n_rows - 1}")
        next_row = min(row + 1, n_rows - 1)  # 아래로 이동할 때, 그리드 범위를 벗어나지 않도록 제한
        next_col = col
        print(f"next_row : {next_row} , next_col : {next_col}")

    elif action == 'left':
        print(f"case 3 의 col-1={col - 1}")
        next_row = row
        next_col = max(col - 1, 0)  # 왼쪽으로 이동할 때, 그리드 범위를 벗어나지 않도록 제한
        print(f"next_row : {next_row} , next_col : {next_col}")

    elif action == 'right':
        print(f"case 4 의 col+1={col+1} , n_cols-1={n_cols - 1}")
        next_row = row
        next_col = min(col + 1, n_cols - 1)  # 오른쪽으로 이동할 때, 그리드 범위를 벗어나지 않도록 제한
        print(f"next_row : {next_row} , next_col : {next_col}")

    # 다음 상태 계산 (좌표를 상태 번호로 변환)
    next_state = next_row * n_cols + next_col
    reward = -1  # 모든 이동의 비용은 -1로 설정 (즉, 이동은 항상 비용이 든다)
    # reward = 10 if next_state == goal_points - 1 else -1  # 목표 상태 도착 시 보상은 1, 나머지는 -1

    print(f"next_state: {next_state} \t reward: {reward}")
    print("---------------------- transition_reward 함수 종료 ----------------------")

    return next_state, reward

# 애니메이션을 위한 데이터 저장
frames = []
iteration_counts = []
convergence_flags = []

# 정책 평가 함수
# 정책 평가(policy evaluation): 주어진 정책에 따라 각 상태의 가치를 계산하는 과정입니다. 주어진 정책을 따른다고 가정했을 때, 각 상태에서 기대되는 장기적인 보상의 총합을 계산합니다.
# 이 과정은 정책의 품질을 평가하는 데 사용되며, 정책을 개선하기 위한 기초 정보를 제공합니다.
# 정책 평가를 통해 얻어진 상태 가치 함수 V는 정책이 얼마나 좋은지를 수치적으로 나타내며, 특정 상태에서 어떤 행동을 선택하는 것이 더 유리한지를 평가하는 데 도움이 됩니다.
def policy_evaluation(policy, gamma=0.9, theta=1e-6):
    k = 0  # iteration 카운트
    while True:
        delta = 0
        # 모든 상태에 대해 가치 업데이트
        for state in range(n_states):
            v_new = 0
            # 현재 상태에서 가능한 모든 행동에 대해 기대 가치 계산
            for action_idx, action_prob in enumerate(policy[state]):
                next_state, reward = transition_reward(state, actions[action_idx])
                v_new += action_prob * (reward + gamma * v[next_state])
            delta = max(delta, abs(v_new - v[state]))
            v_new_states[state] = v_new

        # 가치 함수 동기적 업데이트
        v[:] = v_new_states[:]
        
        # 애니메이션 프레임 저장 (모든 상태가 업데이트된 후 저장)
        frames.append(v.copy())
        iteration_counts.append(k)
        convergence_flags.append(delta < theta)

        # 변화량이 충분히 작으면 (theta 이하이면) 종료
        if delta < theta:
            break
        k += 1  # iteration 증가

    return v

# 정책 평가 실행
v_pi = policy_evaluation(policy, gamma)

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(6, 6))

def update(frame_idx):
    ax.clear()
    v = frames[frame_idx]
    iteration = iteration_counts[frame_idx]
    converged = convergence_flags[frame_idx]
    ax.imshow(v.reshape(n_rows, n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=0)
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

# 코드 설명:
# - 이 코드는 강화학습의 정책 평가를 수행하는 예제입니다. 4x4 그리드 환경에서 상태 가치 함수 V를 업데이트하며, 랜덤 정책에 따라 각 상태에서 네 가지 행동(up, down, left, right)을 같은 확률로 선택합니다.
# - 상태 전이 함수(transition_reward)는 현재 상태와 선택한 행동에 따라 다음 상태와 보상을 반환하며, 모든 이동은 보상이 -1로 설정됩니다.
# - 정책 평가(policy evaluation) 함수는 가치 함수 V를 반복적으로 업데이트하여 정책이 주어진 상황에서 얼마나 좋은지를 평가합니다.
# - 각 상태 업데이트 후 애니메이션 프레임을 저장하고, 이를 통해 반복(iteration)마다 그리드의 가치를 시각적으로 보여주는 애니메이션을 생성합니다.
# - gamma는 할인율로 미래의 보상을 현재 가치로 얼마나 할인할지 결정하며, theta는 수렴 조건으로 가치 변화량이 충분히 작으면 반복을 종료합니다.
# - 정책 평가를 수행하는 이유: 정책 평가를 통해 현재 주어진 정책(policy)이 각 상태에서 얼마나 좋은지를 나타내는 가치 함수 V를 구할 수 있습니다. 이를 통해 특정 정책 하에서 기대되는 장기적인 보상(return)을 알 수 있으며, 정책의 품질을 평가하거나, 이후 정책 개선(policy improvement)을 위한 기반 정보를 제공할 수 있습니다.
# - 예상 결과: 랜덤 정책은 상하좌우로 동일한 확률(0.25)로 움직이기 때문에, 모든 상태의 가치가 균일하게 분포하는 경향이 있습니다. 이는 특정 목표 상태로 빠르게 이동하는 최적의 정책과는 다르며, 따라서 각 상태의 가치는 -1로 시작해 모든 방향으로의 이동 비용을 균일하게 반영한 값이 됩니다. 최적의 정책이 아닐 경우, 가치 함수는 전반적으로 낮은 값을 가지며, 특정 상태로의 수렴이 더디게 일어납니다.
# - 위 결과에서 모든 상태의 가치가 거의 동일한 값을 가진 이유는 정책이 랜덤이기 때문에 각 상태에서 특정 목표로 향하는 방향성이 없이 모든 방향으로 동일한 확률로 움직이기 때문입니다.
# - 즉, 정책이 최적화되지 않은 상태에서는 각 상태의 가치는 균일하게 분포하는 경향이 있습니다. 모든 상태에서 각 행동이 동일한 확률로 선택되므로, 가치 함수는 이동할 때마다 일정한 비용(-1)을 계속 더해지는 구조를 가집니다. 결과적으로 모든 상태에서의 기대 보상(return)은 비슷하게 수렴하게 됩니다.
# - 이러한 상황에서 각 상태의 가치가 거의 동일하다는 것은 랜덤 정책이 특정 목표 상태로 이동하는 데 있어 효과적이지 않으며, 모든 상태에서 동일하게 랜덤하게 행동한다는 의미를 갖습니다.
# - 따라서 정책 개선을 통해 목표 상태로 더 빠르게 도달할 수 있는 방향성을 가지도록 변경할 필요가 있습니다.