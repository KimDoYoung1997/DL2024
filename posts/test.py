import numpy as np
import matplotlib.pyplot as plt
import time

# 4x4 그리드 환경 정의 (16개의 상태)
n_rows, n_cols = 4, 4
n_states = n_rows * n_cols

# 행동 정의 (상, 하, 좌, 우)
actions = ['up', 'down', 'left', 'right']
# 할인율b
gamma = 0.9
# 상태 가치 함수 초기화 (모든 상태의 가치를 0으로 시작)
v = np.zeros(n_states)
# 정책 정의 (랜덤 정책으로 설정) -> 모든 상태에서 네 방향으로 같은 확률로 이동
policy = np.ones((n_states, len(actions))) / len(actions)

# 상태 전이 확률 및 보상
def transition_reward(state, action):
    row, col = divmod(state, n_cols)  # 상태를 그리드 좌표로 변환
    print("---------------------- transition_reward 함수 실행 ----------------------")
    print(f"row : {row}, col : {col}")
    if action == 'up':
        print(f"case 1 의 row-1={row-1}")
        next_row = max(row - 1, 0)
        next_col = col
        print(f"next_row : {next_row} , next_col : {next_col}")

    elif action == 'down':
        print(f"case 2 의 row+1={row+1} , n_rows-1={n_rows - 1}")
        next_row = min(row + 1, n_rows - 1) # n_rows = 4, row =
        next_col = col
        print(f"next_row : {next_row} , next_col : {next_col}")
    elif action == 'left':
        print(f"case 3 의 col-1={col - 1}")
        next_row = row
        next_col = max(col - 1, 0)
        print(f"next_row : {next_row} , next_col : {next_col}")

    elif action == 'right':
        print(f"case 4 의 col+1={col+1} , n_cols-1={n_cols - 1}")
        next_row = row
        next_col = min(col + 1, n_cols - 1)
        print(f"next_row : {next_row} , next_col : {next_col}")

    next_state = next_row * n_cols + next_col
    reward = -1  # 모든 이동의 비용은 -1로 설정
    print(f"next_state: {next_state} \t reward: {reward}")
    print("---------------------- transition_reward 함수 종료 ----------------------")

    return next_state, reward

# 미로 그리드 시각화 함수
def plot_grid(v, k, state=None, action=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(v.reshape(n_rows, n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=0)
    plt.colorbar(label='Value')
    plt.title(f'Iteration: {k}, State: {state}, Action: {action}')

    for i in range(n_rows):
        for j in range(n_cols):
            state_value = v[i * n_cols + j]
            plt.text(j, i, f'{state_value:.1f}', ha='center', va='center', color='black')

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.pause(1)  # 1초 대기 (각 상태 변화를 쉽게 확인)
    plt.show()


# 정책 평가 함수
def policy_evaluation(policy, gamma=0.9, theta=1e-6):
    k = 0  # iteration 카운트
    while True:
        delta = 0
        # 모든 상태에 대해 가치 업데이트
        for state in range(n_states):
            v_new = 0
            # 현재 상태에서 가능한 모든 행동에 대해 기대 가치 계산
            # policy[state] 는 len이 4인 array ,  array([0.25, 0.25, 0.25, 0.25]) up down left right
            for action_idx, action_prob in enumerate(policy[state]):
                # 행동에 따라 다음 상태와 보상 계산
                next_state, reward = transition_reward(state, actions[action_idx])
                # 가치 함수 업데이트: 보상 + 할인된 다음 상태 가치
                v_new += action_prob * (reward + gamma * v[next_state]) 
                # state 0에서 up 할때  next_state:0 으로 전이되고 , reward:-1을 받는다. state 0에서 up 한거에 대한 value는  -0.25 == 0.25 * (-1 + 0.9*0) 이며 , 
                    # v_new = 0 + (-0.25)
                # state 0에서 down 할때 next_state:4 으로 전이되고, reward:-1을 받는다. state 0에서 down 한거에 대한 value는 -0.25 == 0.25 * (-1 + 0.9*0) 이며,   
                    # v_new = 0+ (-0.25) +(-0.25)
                # state 0에서 left 할때 next_state:0 으로 전이되고, reward:-1을 받는다. state 0에서 left 한거에 대한 value는 -0.25 == 0.25 * (-1 + 0.9*0) 이며,   
                    # v_new = 0+(-0.25)+(-0.25)+(-0.25)
                # state 0에서 right 할때 next_state:1 으로 전이되고, reward:-1을 받는다. state 0에서 right 한거에 대한 value는 -0.25 == 0.25 * (-1 + 0.9*0) 이며,   
                    # v_new = 0+(-0.25)+(-0.25)+(-0.25)+(-0.25)
                # 이제 안쪽 for문에서 탈출한다. state 에 대한 모든 action을 다 더해 v_new에 대한 기대값을 계산 완료했기 때문이다.

                # state 1에서 up 할때  next_state:0 으로 전이되고 , reward:-1을 받는다. state 0에서 up 한거에 대한 value는  -0.25 == 0.25 * (-1 + 0.9*0) 이며 , 
                    # v_new = 0 + (-0.25)
                # state 0에서 down 할때 next_state:4 으로 전이되고, reward:-1을 받는다. state 0에서 down 한거에 대한 value는 -0.25 == 0.25 * (-1 + 0.9*0) 이며,   
                    # v_new = 0+ (-0.25) +(-0.25)
                # state 0에서 left 할때 next_state:0 으로 전이되고, reward:-1을 받는다. state 0에서 left 한거에 대한 value는 -0.25 == 0.25 * (-1 + 0.9*0) 이며,   
                    # v_new = 0+(-0.25)+(-0.25)+(-0.25)
                # state 0에서 right 할때 next_state:1 으로 전이되고, reward:-1을 받는다. state 0에서 right 한거에 대한 value는 -0.25 == 0.25 * (-1 + 0.9*0) 이며,   
                    # v_new = 0+(-0.25)+(-0.25)+(-0.25)+(-0.25)
                # state 1에서 left 할때 next_state:1 으로 전이되고, reward:-1을 받는다. state 1에서 left 한거에 대한 value는 -0.475 == 0.25 * (-1 + 0.9*(-1.0)) 이며,   
                    # v_new = 0+(-0.25)+(-0.25)+(-0.475) = -0.975
            # 최대 변화량을 기록 (수렴 여부 확인을 위해)
            delta = max(delta, abs(v_new - v[state]))       # state 0 에서의 v 값인 v[state==0] 은 기존에 0, v_new = -1, delta=0 , 
                                                            # delta = max(0,abs(-1-0))=1 으로 재정의후 v[state==0] 또한 v_new 으로 재정의
                                                            # state 1 에서 delta = max(1,abs(-1.225-0))=1.225
            v[state] = v_new  # 상태 가치 함수 업데이트 

            # 미로 그리드를 업데이트하고 상태, 행동을 시각적으로 확인
            plot_grid(v, k, state, actions[action_idx])

        # 변화량이 충분히 작으면 종료
        if delta < theta:
            break
        k += 1  # iteration 증가

    return v

v_pi = policy_evaluation(policy, gamma)



# import numpy as np
# import matplotlib.pyplot as plt
# import time

# # 4x4 그리드 환경 정의 (16개의 상태)
# n_rows, n_cols = 4, 4
# n_states = n_rows * n_cols

# # 행동 정의 (상, 하, 좌, 우)
# actions = ['up', 'down', 'left', 'right']

# # 정책 정의 (랜덤 정책으로 설정) -> 모든 상태에서 네 방향으로 같은 확률로 이동
# policy = np.ones((n_states, len(actions))) / len(actions)

# # 상태 전이 확률 및 보상
# def transition_reward(state, action):
#     row, col = divmod(state, n_cols)  # 상태를 그리드 좌표로 변환
#     if action == 'up':
#         next_row = max(row - 1, 0)
#         next_col = col
#     elif action == 'down':
#         next_row = min(row + 1, n_rows - 1)
#         next_col = col
#     elif action == 'left':
#         next_row = row
#         next_col = max(col - 1, 0)
#     elif action == 'right':
#         next_row = row
#         next_col = min(col + 1, n_cols - 1)

#     next_state = next_row * n_cols + next_col
#     reward = -1  # 모든 이동의 비용은 -1로 설정
#     return next_state, reward

# # 할인율
# gamma = 0.9

# # 상태 가치 함수 초기화 (모든 상태의 가치를 0으로 시작)
# v = np.zeros(n_states)

# # 미로 그리드 시각화 함수
# def plot_grid(v, k, state=None, action=None):
#     plt.figure(figsize=(6, 6))
#     plt.imshow(v.reshape(n_rows, n_cols), cmap='coolwarm', interpolation='none', vmin=-20, vmax=0)
#     plt.colorbar(label='Value')
#     plt.title(f'Iteration: {k}, State: {state}, Action: {action}')

#     for i in range(n_rows):
#         for j in range(n_cols):
#             state_value = v[i * n_cols + j]
#             plt.text(j, i, f'{state_value:.1f}', ha='center', va='center', color='black')

#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.pause(1)  # 1초 대기 (각 상태 변화를 쉽게 확인)
#     plt.show()

# # 정책 평가 함수
# def policy_evaluation(policy, gamma=0.9, theta=1e-6):
#     k = 0  # iteration 카운트
#     while True:
#         delta = 0
#         # 모든 상태에 대해 가치 업데이트
#         for state in range(n_states):
#             v_new = 0
#             # 현재 상태에서 가능한 모든 행동에 대해 기대 가치 계산
#             for action_idx, action_prob in enumerate(policy[state]):
#                 # 행동에 따라 다음 상태와 보상 계산
#                 next_state, reward = transition_reward(state, actions[action_idx])
#                 # 가치 함수 업데이트: 보상 + 할인된 다음 상태 가치
#                 v_new += action_prob * (reward + gamma * v[next_state])

#             # 최대 변화량을 기록 (수렴 여부 확인을 위해)
#             delta = max(delta, abs(v_new - v[state]))
#             v[state] = v_new  # 상태 가치 함수 업데이트

#             # 미로 그리드를 업데이트하고 상태, 행동을 시각적으로 확인
#             plot_grid(v, k, state, actions[action_idx])

#         # 변화량이 충분히 작으면 종료
#         if delta < theta:
#             break
#         k += 1  # iteration 증가

#     return v

# # 정책 평가 실행
# v_pi = policy_evaluation(policy, gamma)
