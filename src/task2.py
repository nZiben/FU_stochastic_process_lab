import numpy as np
import matplotlib.pyplot as plt

def step_markov_chain(P, p0, steps=1):
    """
    Делает один (или несколько) шагов марковской цепи.
    P  - матрица переходов (size = (m, m))
    p0 - вектор распределения (size = (m,))
    """
    p_current = p0
    for _ in range(steps):
        p_current = p_current @ P
    return p_current

def simulate_one_trajectory(P, p0, length=1000, random_seed=42):
    """
    Генерация одной траектории длины length.
    """
    np.random.seed(random_seed)
    m = P.shape[0]
    # Изначально состояние выбирается из p0
    states = []
    initial_state = np.random.choice(np.arange(m), p=p0)
    states.append(initial_state)
    
    for i in range(1, length):
        current_state = states[-1]
        next_state = np.random.choice(np.arange(m), p=P[current_state])
        states.append(next_state)
    
    return np.array(states)

def estimate_transition_matrix(trajectory, m):
    """
    Оцениваем матрицу перехода по одной траектории.
    """
    count_matrix = np.zeros((m, m))
    
    for i in range(len(trajectory)-1):
        count_matrix[trajectory[i], trajectory[i+1]] += 1
    
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    # чтобы избежать деления на 0 (если в траектории не встретилось какое-то состояние)
    row_sums[row_sums == 0] = 1  
    P_est = count_matrix / row_sums
    return P_est

def main():
    # Матрица переходов (P) и начальное распределение (p) (пример варианта)
    P = np.array([
        [73/334,  99/334,  0,       0,       83/334,  79/334 ],
        [17/62,   0,       89/248,  91/248,  0,       0      ],
        [0,       94/335,  69/335,  0,       20/67,   72/335 ],
        [17/88,   83/440,  5/22,    0,       21/110,  1/5    ],
        [59/325,  0,       87/325,  4/13,    79/325,  0      ],
        [39/164,  35/164,  0,       35/164,  55/164,  0      ]
    ])
    
    p0 = np.array([76/291, 0, 23/97, 53/291, 15/97, 16/97])
    
    # 1) Три траектории
    plt.figure(figsize=(8, 5))
    for i in range(3):
        traj = simulate_one_trajectory(P, p0, length=50, random_seed=42+i)
        plt.plot(traj, label=f"Траектория {i+1}")
    plt.legend()
    plt.title("Три траектории марковской цепи")
    plt.xlabel("t")
    plt.ylabel("Состояние")
    plt.grid(True)
    plt.show()
    
    # 2) Распределение на шаге k=29
    p_k = step_markov_chain(P, p0, steps=29)
    print("Распределение на шаге k=29 (округление до 6 знаков):")
    print(np.round(p_k, 6))
    
    # 3) Смоделируем одну большую траекторию и оценим P
    big_traj = simulate_one_trajectory(P, p0, length=10_000, random_seed=123)
    P_est_big = estimate_transition_matrix(big_traj, m=P.shape[0])
    print("Оцененная матрица перехода по одной большой траектории:")
    print(P_est_big)
    
    # 4) Симулируем N=300 траекторий и оценим P, нач.распр, распр на шаге k
    N = 300
    all_traj = []
    for i in range(N):
        tr = simulate_one_trajectory(P, p0, length=50, random_seed=1000+i)
        all_traj.append(tr)
    
    # Оценка P
    # Соберём все переходы вместе
    all_states = np.concatenate(all_traj)
    P_est_300 = estimate_transition_matrix(all_states, m=P.shape[0])
    
    # Оценка начального распределения (по состояниям на шаге 0)
    first_states = np.array([traj[0] for traj in all_traj])
    counts_first = np.bincount(first_states, minlength=P.shape[0])
    p0_est_300 = counts_first / N
    
    # Оценка распределения на шаге k=29 (берём 29-й элемент каждой траектории)
    # если длина 50, тогда индекс 29 доступен
    step_k_states = np.array([traj[29] for traj in all_traj])
    counts_k = np.bincount(step_k_states, minlength=P.shape[0])
    p_k_est_300 = counts_k / N
    
    print("Оценка P по 300 траекториям:")
    print(P_est_300)
    print("Оценка начального распределения по 300 траекториям:")
    print(p0_est_300)
    print(f"Оценка распределения на шаге k=29 по 300 траекториям:")
    print(p_k_est_300)
    
    # 5) Стационарные распределения (если существуют)
    # Для эргодических цепей решение системы p = pP, sum(p)=1
    # Здесь просто пример численного решения
    eigvals, eigvecs = np.linalg.eig(P.T)
    # Находим собственный вектор с собственным значением = 1
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi_stationary = np.real(eigvecs[:, idx])
    pi_stationary = pi_stationary / np.sum(pi_stationary)
    print("Стационарное распределение (численно найденное):")
    print(pi_stationary)
    
    # 6) Можно построить графики p_{ij}(n) и т.д. - здесь набросок кода
    # 7) Тест хи-квадрат на независимость значений цепи на шагах k и 2k
    #    (требуется аккуратная реализация)

if __name__ == "__main__":
    main()
