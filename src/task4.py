import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, chisquare
import csv

def simulate_poisson_process(lmbd, T, random_seed=None):
    """
    Симуляция процесса Пуассона с интенсивностью lmbd на интервале [0, T].

    Возвращает:
    - events: массив времен событий
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    events = []
    t = 0.0
    while True:
        inter_arrival = np.random.exponential(scale=1/lmbd)
        t += inter_arrival
        if t > T:
            break
        events.append(t)
    return np.array(events)

def run_simulations(lmbd, T, N_simulations, random_seed=None):
    """
    Проведение N_simulations симуляций процесса Пуассона.

    Возвращает:
    - counts: список количества событий в каждой симуляции
    """
    counts = []
    for i in range(N_simulations):
        seed = random_seed + i if random_seed is not None else None
        events = simulate_poisson_process(lmbd, T, random_seed=seed)
        counts.append(len(events))
    return counts

def compute_N_t_estimates(counts, lmbd, T):
    """
    Вычисление оценок N_t^ и N_t^∘.

    Параметры:
    - counts: список количества событий в каждой симуляции
    - lmbd: интенсивность процесса
    - T: момент времени t

    Возвращает:
    - N_t_hat: список оценок N_t^
    - N_t_circ: список оценок N_t^∘
    """
    # Предполагаем, что s = T/2 для упрощения
    s = T / 2
    N_t_hat = [count + lmbd * (T - s) for count in counts]
    N_t_circ = [count * (T / s) for count in counts]
    return N_t_hat, N_t_circ

def main():
    # Параметры варианта (например, n=17 => lambda=√17)
    n = 17
    lmbd = np.sqrt(n)
    T = n  # момент времени t=n
    print(f"Параметры варианта:\nlambda={lmbd:.6f}, T={T}\n")

    # 1. Симуляция процесса Пуассона
    events = simulate_poisson_process(lmbd, T, random_seed=42)
    N_n = len(events)
    print(f"Число событий к моменту t={T}: {N_n}")

    # Сохранение событий в файл
    np.savetxt("task4_events.csv", events, delimiter=',', header="Event Times", comments='')

    # 2. Проверка гипотезы о распределении Пуассона
    # Проведём множественные симуляции и сравним эмпирическое распределение с теорией
    N_simulations = 1000
    counts = run_simulations(lmbd, T, N_simulations, random_seed=100)
    lambda_poisson = lmbd * T
    expected_counts = poisson.pmf(k=np.arange(max(counts)+1), mu=lambda_poisson) * N_simulations

    # Эмпирические частоты
    empirical_counts, bin_edges = np.histogram(counts, bins=range(max(counts)+2), density=False)

    # Вычисление статистики хи-квадрат
    # Убедимся, что все ожидаемые частоты >= 5
    mask = expected_counts >= 5
    sum_obs = empirical_counts[mask].sum()
    sum_exp = expected_counts[mask].sum()
    scale_factor = sum_obs / sum_exp
    scaled_expected = expected_counts[mask] * scale_factor

    # Корректировка последнего бинна для точного совпадения сумм
    difference = sum_obs - scaled_expected.sum()
    scaled_expected[-1] += difference

    chi2_stat, p_value = chisquare(f_obs=empirical_counts[mask], f_exp=scaled_expected)
    print(f"Критерий хи-квадрат:\nСтатистика: {chi2_stat:.4f}\np-value: {p_value:.4f}")

    # Сохранение результатов симуляций
    with open("task4_simulation_counts.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Simulation", "Event Count"])
        for i, count in enumerate(counts):
            writer.writerow([i+1, count])

    # 3. Оценки N_t^ и N_t^∘
    N_t_hat, N_t_circ = compute_N_t_estimates(counts, lmbd, T)
    mse_hat = np.mean((np.array(N_t_hat) - (lmbd * T))**2)
    mse_circ = np.mean((np.array(N_t_circ) - (lmbd * T))**2)
    print(f"\nОценки и MSE:")
    print(f"MSE для N_t^: {mse_hat:.4f}")
    print(f"MSE для N_t^∘: {mse_circ:.4f}")

    # Сохранение оценок в файл
    with open("task4_N_t_estimates.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Simulation", "N_t_hat", "N_t_circ"])
        for i in range(N_simulations):
            writer.writerow([i+1, N_t_hat[i], N_t_circ[i]])

    # 4. Визуализация распределения количества событий
    plt.figure(figsize=(10,6))
    plt.hist(counts, bins=range(max(counts)+2), density=True, alpha=0.6, color='g', label='Эмпирическое')
    plt.plot(np.arange(max(counts)+1), poisson.pmf(k=np.arange(max(counts)+1), mu=lambda_poisson),
             'ro-', label='Теоретическое')
    plt.xlabel('Число событий')
    plt.ylabel('Относительная частота')
    plt.title('Сравнение эмпирического и теоретического распределения Пуассона')
    plt.legend()
    plt.savefig("task4_poisson_distribution_comparison.png")
    plt.show()

    # 5. Визуализация MSE
    plt.figure(figsize=(6,4))
    plt.bar(['N_t^', 'N_t^∘'], [mse_hat, mse_circ], color=['blue', 'orange'])
    plt.ylabel('MSE')
    plt.title('Среднеквадратичные отклонения оценок')
    plt.savefig("task4_MSE_comparison.png")
    plt.show()

    # 6. Вывод результатов в файл
    with open("task4_report_summary.txt", "w") as f:
        f.write(f"Параметры:\nlambda={lmbd:.6f}, T={T}\n")
        f.write(f"Число событий к моменту t={T}: {N_n}\n\n")
        f.write(f"Критерий хи-квадрат:\nСтатистика: {chi2_stat:.4f}\np-value: {p_value:.4f}\n\n")
        f.write(f"Среднеквадратичные отклонения:\nMSE для N_t^: {mse_hat:.4f}\nMSE для N_t^∘: {mse_circ:.4f}\n")

if __name__ == "__main__":
    main()
