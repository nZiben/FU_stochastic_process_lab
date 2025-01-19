import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import os
from scipy.stats import chisquare
from scipy.linalg import fractional_matrix_power

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
    states = []
    initial_state = np.random.choice(np.arange(m), p=p0)
    states.append(initial_state)
    
    for _ in range(1, length):
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
    # Чтобы избежать деления на 0 (если в траектории не встретилось какое-то состояние)
    row_sums[row_sums == 0] = 1  
    P_est = count_matrix / row_sums
    return P_est

def plot_trajectories(trajectories, filepath):
    """
    Построение и сохранение графика траекторий Марковской цепи.
    
    Параметры:
    - trajectories: список массивов траекторий
    - filepath: путь для сохранения графика
    """
    plt.figure(figsize=(12, 8))
    for i, traj in enumerate(trajectories, 1):
        plt.plot(traj, label=f"Траектория {i}")
    plt.legend()
    plt.title("Траектории Марковской цепи")
    plt.xlabel("Шаг (n)")
    plt.ylabel("Состояние")
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    print(f"Траектории Марковской цепи сохранены в {filepath}")

def plot_transition_probabilities(P_powers, states, filepath):
    """
    Построение графиков вероятностей перехода p_ij(n) от шага n.
    
    Параметры:
    - P_powers: массив матриц P^n для различных n
    - states: список кортежей состояний (i, j) для построения графиков
    - filepath: путь для сохранения графика
    """
    plt.figure(figsize=(12, 8))
    
    n_values = np.arange(1, len(P_powers)+1)
    
    for (i, j) in states:
        p_ij = [P_powers[n-1][i, j] for n in n_values]
        plt.plot(n_values, p_ij, label=f"p_{{{i}{j}}}(n)")
    
    plt.title("Переходные вероятности p_{ij}(n) от шага n")
    plt.xlabel("Шаг (n)")
    plt.ylabel("Вероятность p_{ij}(n)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    print(f"Графики переходных вероятностей сохранены в {filepath}")

def chi_square_test_independence(states_k, states_2k, filepath):
    """
    Проведение теста \(\chi^2\) на независимость состояний на шагах k и 2k.
    
    Параметры:
    - states_k: массив состояний на шаге k
    - states_2k: массив состояний на шаге 2k
    - filepath: путь для сохранения результатов теста
    """
    # Создание таблицы сопряженности
    contingency_table = np.zeros((states_k.max()+1, states_2k.max()+1), dtype=int)
    for s1, s2 in zip(states_k, states_2k):
        contingency_table[s1, s2] += 1
    
    # Проведение теста \(\chi^2\)
    chi2_stat, p_value = chisquare(contingency_table, axis=None)
    
    # Сохранение результатов теста в файл с указанием кодировки utf-8
    with open(filepath, "w", encoding='utf-8') as f:
        f.write("Тест согласия χ² на независимость состояний на шагах k и 2k\n")
        f.write("Таблица сопряженности:\n")
        f.write(" " + " ".join([f"{j}" for j in range(contingency_table.shape[1])]) + "\n")
        for i, row in enumerate(contingency_table):
            f.write(f"{i} " + " ".join(map(str, row)) + "\n")
        f.write(f"\nСтатистика χ²: {chi2_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
    
    print(f"Результаты теста χ² на независимость сохранены в {filepath}")
    return chi2_stat, p_value

def compute_stationary_distribution(P):
    """
    Вычисление стационарного распределения как собственный вектор матрицы P^T при λ=1.
    
    Параметры:
    - P: матрица переходов (size = (m, m))
    
    Возвращает:
    - pi_stationary: стационарное распределение (вектор)
    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    # Находим собственный вектор с собственным значением = 1
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi_stationary = np.real(eigvecs[:, idx])
    pi_stationary = pi_stationary / np.sum(pi_stationary)
    return pi_stationary

def main():
    # Создание директории для графиков, если не существует
    os.makedirs("charts", exist_ok=True)
    
    # Матрица переходов (P) и начальное распределение (p0)
    P = np.array([
        [73/334,  99/334,  0,       0,       83/334,  79/334 ],
        [17/62,   0,       89/248,  91/248,  0,       0      ],
        [0,       94/335,  69/335,  0,       20/67,   72/335 ],
        [17/88,   83/440,  5/22,    0,       21/110,  1/5    ],
        [59/325,  0,       87/325,  4/13,    79/325,  0      ],
        [39/164,  35/164,  0,       35/164,  55/164,  0      ]
    ])
    
    p0 = np.array([76/291, 0, 23/97, 53/291, 15/97, 16/97])
    
    # 1) Генерация 3 траекторий длиной 50 шагов каждая
    trajectories = []
    for i in range(3):
        traj = simulate_one_trajectory(P, p0, length=50, random_seed=42+i)
        trajectories.append(traj)
    
    # Построение и сохранение графика траекторий
    plot_trajectories(trajectories, filepath="charts/task2_trajectories.png")
    
    # 2) Распределение на шаге k=29
    k = 29
    p_k = step_markov_chain(P, p0, steps=k)
    print(f"Распределение на шаге k={k} (округление до 6 знаков):")
    print(np.round(p_k, 6))
    
    # Сохранение распределения на шаге k=29 в текстовый файл
    with open("charts/task2_p_k_29.txt", "w", encoding='utf-8') as f:
        f.write(f"Распределение на шаге k={k}:\n")
        for state, prob in enumerate(p_k):
            f.write(f"Состояние {state}: {prob:.6f}\n")
    print(f"Распределение на шаге k={k} сохранено в charts/task2_p_k_29.txt")
    
    # 3) Симуляция одной большой траектории длиной 10,000 шагов и оценка матрицы P
    big_traj = simulate_one_trajectory(P, p0, length=10_000, random_seed=123)
    P_est_big = estimate_transition_matrix(big_traj, m=P.shape[0])
    print("Оцененная матрица переходов по одной большой траектории:")
    print(P_est_big)
    
    # Сохранение оцененной матрицы P
    np.savetxt("charts/task2_P_est_big.csv", P_est_big, delimiter=',', header="P_est_big", comments='')
    print("Оцененная матрица переходов (P_est_big) сохранена в charts/task2_P_est_big.csv")
    
    # 4) Симуляция 300 траекторий по 50 шагов и оценка P, p0, p(k=29)
    N = 300
    traj_length = 50
    all_traj = []
    for i in range(N):
        tr = simulate_one_trajectory(P, p0, length=traj_length, random_seed=1000+i)
        all_traj.append(tr)
    
    # Оценка матрицы переходов по 300 траекториям
    all_states = np.concatenate(all_traj)
    P_est_300 = estimate_transition_matrix(all_states, m=P.shape[0])
    print("Оцененная матрица переходов по 300 траекториям:")
    print(P_est_300)
    
    # Сохранение оцененной матрицы P
    np.savetxt("charts/task2_P_est_300.csv", P_est_300, delimiter=',', header="P_est_300", comments='')
    print("Оцененная матрица переходов (P_est_300) сохранена в charts/task2_P_est_300.csv")
    
    # Оценка начального распределения p0 по 300 траекториям
    first_states = np.array([traj[0] for traj in all_traj])
    counts_first = np.bincount(first_states, minlength=P.shape[0])
    p0_est_300 = counts_first / N
    print("Оцененное начальное распределение по 300 траекториям:")
    print(p0_est_300)
    
    # Сохранение оцененного p0
    with open("charts/task2_p0_est_300.txt", "w", encoding='utf-8') as f:
        f.write("Оцененное начальное распределение (p0_est_300):\n")
        for state, prob in enumerate(p0_est_300):
            f.write(f"Состояние {state}: {prob:.6f}\n")
    print("Оцененное начальное распределение (p0_est_300) сохранено в charts/task2_p0_est_300.txt")
    
    # Оценка распределения на шаге k=29 по 300 траекториям
    if traj_length > k:
        step_k_states = np.array([traj[k] for traj in all_traj])
        counts_k = np.bincount(step_k_states, minlength=P.shape[0])
        p_k_est_300 = counts_k / N
        print(f"Оцененное распределение на шаге k={k} по 300 траекториям:")
        print(p_k_est_300)
        
        # Сохранение оцененного распределения на шаге k=29
        with open("charts/task2_p_k_29_est_300.txt", "w", encoding='utf-8') as f:
            f.write(f"Оцененное распределение на шаге k={k} по 300 траекториям:\n")
            for state, prob in enumerate(p_k_est_300):
                f.write(f"Состояние {state}: {prob:.6f}\n")
        print(f"Оцененное распределение на шаге k={k} по 300 траекториям сохранено в charts/task2_p_k_29_est_300.txt")
    else:
        print(f"Траектории слишком короткие для оценки состояния на шаге k={k}.")
    
    # 5) Вычисление стационарного распределения
    pi_stationary = compute_stationary_distribution(P)
    print("Стационарное распределение (pi_stationary):")
    print(pi_stationary)
    
    # Сохранение стационарного распределения
    with open("charts/task2_pi_stationary.txt", "w", encoding='utf-8') as f:
        f.write("Стационарное распределение (pi_stationary):\n")
        for state, prob in enumerate(pi_stationary):
            f.write(f"Состояние {state}: {prob:.6f}\n")
    print("Стационарное распределение (pi_stationary) сохранено в charts/task2_pi_stationary.txt")
    
    # 6) Построение графиков переходных вероятностей p_{ij}(n) от шага n
    # Вычислим P^n для n от 1 до 100
    max_n = 100
    P_powers = [np.linalg.matrix_power(P, n) for n in range(1, max_n+1)]
    
    # Выберем несколько пар состояний для построения графиков
    selected_states = [(0,0), (0,1), (1,2), (2,5), (3,4), (4,0), (5,1)]
    
    plot_transition_probabilities(P_powers, selected_states, filepath="charts/task2_transition_probabilities.png")
    
    # 7) Тест \(\chi^2\) на независимость состояний на шагах k и 2k
    # Выберем k=29, 2k=58
    k2 = 2 * k
    # Проверим, что траектории достаточно длинные
    if traj_length >= k2 + 1:
        states_k = np.array([traj[k] for traj in all_traj])
        states_2k = np.array([traj[k2] for traj in all_traj])
        
        chi2_stat, p_value = chi_square_test_independence(states_k, states_2k, filepath="charts/task2_chi_square_independence.txt")
        print(f"\nРезультаты теста χ² на независимость состояний на шагах k={k} и 2k={k2}:")
        print(f"Статистика χ²: {chi2_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
    else:
        print(f"Траектории слишком короткие для проведения теста на шагах k={k} и 2k={k2}.")
    
    # Дополнительно: Вывод результатов можно сохранить в отчетные файлы
    # Например, матрицу P_est_300, p0_est_300 и p_k_est_300 уже сохранены выше
    # Если требуется, можно создать сводную таблицу или дополнительные файлы

if __name__ == "__main__":
    main()
