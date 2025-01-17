import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from collections import deque

def compute_stationary_distribution(k, m, r, l):
    """
    Вычисление теоретического стационарного распределения числа работающих блоков
    для системы с k+2 рабочими и m+k+2 блоками.

    Параметры:
    - k: остаток от деления номера варианта на 4
    - m: остаток от деления номера варианта на 7
    - r: остаток от деления (n + k) на 3
    - l: остаток от деления (n + m) на 5

    Возвращает:
    - pi: массив стационарных вероятностей для каждого состояния (от 0 до m+k+2)
    """
    # Количество состояний: от 0 до m+k+2
    num_states = m + k + 2 + 1
    A = np.zeros((num_states, num_states))
    b = np.zeros(num_states)

    # Заполняем матрицу коэффициентов A и правую часть b для системы pi * Q = 0
    # Где Q - матрица генераторов
    for i in range(num_states):
        if i < k + 2:
            A[i, i] = r + 1  # скорость отказа
            if i + 1 < num_states:
                A[i, i + 1] = 1  # переход из состояния i в i+1 при отказе
        else:
            # Если рабочие заняты, новые отказы идут в очередь
            A[i, i] += r + 1  # скорость отказа

        if i > 0:
            A[i, i - 1] += (k + 2) * (l + 2)  # скорость ремонта
            A[i, i] -= (k + 2) * (l + 2)  # исходящая скорость

        if i == 0:
            A[i, i] = 1  # pi0 = 1 при нормировке

    # Добавляем условие нормировки
    A[-1, :] = 1
    b[-1] = 1

    # Решаем систему A * pi = b
    pi = solve(A, b)

    return pi

def simulate_failure_repair(k, m, r, l, T=1000, random_seed=42):
    """
    Имитация процесса выхода из строя и ремонта блоков машины за время T.

    Параметры:
    - k, m, r, l: параметры варианта
    - T: общее время моделирования
    - random_seed: зерно генератора случайных чисел

    Возвращает:
    - state_history: список состояний системы во времени
    - queue_history: список длин очереди во времени
    - time_points: список временных точек
    """
    np.random.seed(random_seed)
    num_working = m + k + 2  # Изначально все блоки работают
    total_blocks = m + k + 2
    queue = 0  # Начальная очередь
    state_history = []
    queue_history = []
    time_history = []
    current_time = 0.0

    # События: отказ или ремонт
    # Используем приоритетную очередь для событий
    event_queue = deque()

    # Инициализация первого отказа
    time_to_fail = np.random.exponential(scale=1/(r + 1))
    event_queue.append(('fail', current_time + time_to_fail))

    # Инициализация
    state_history.append(num_working)
    queue_history.append(queue)
    time_history.append(current_time)

    while current_time < T and event_queue:
        # Получаем следующее событие
        event = event_queue.popleft()
        event_type, event_time = event

        if event_time > T:
            break

        # Обновляем время
        current_time = event_time

        if event_type == 'fail':
            if num_working > 0:
                num_working -= 1
                if num_working >= (k + 2):
                    queue += 1
                else:
                    # Если есть свободные рабочие, сразу ремонтируем
                    num_working += 1  # Сразу ремонтируем
                    # Добавляем следующее событие ремонта
                    time_to_repair = np.random.exponential(scale=1/(l + 2))
                    event_queue.append(('repair', current_time + time_to_repair))
            else:
                queue += 1

            # Добавляем следующее событие отказа
            time_to_fail = np.random.exponential(scale=1/(r + 1))
            event_queue.append(('fail', current_time + time_to_fail))

        elif event_type == 'repair':
            if queue > 0:
                queue -= 1
                num_working += 1
                # Добавляем следующее событие ремонта
                time_to_repair = np.random.exponential(scale=1/(l + 2))
                event_queue.append(('repair', current_time + time_to_repair))
            else:
                # Нет очереди, рабочий свободен
                pass

        # Записываем состояние
        state_history.append(num_working)
        queue_history.append(queue)
        time_history.append(current_time)

    return state_history, queue_history, time_history

def compute_average_queue(queue_history, time_history):
    """
    Вычисляет среднюю длину очереди за всё время моделирования.

    Параметры:
    - queue_history: список длин очереди во времени
    - time_history: список временных точек

    Возвращает:
    - avg_queue: средняя длина очереди
    """
    total_time = 0.0
    weighted_queue = 0.0
    for i in range(1, len(time_history)):
        dt = time_history[i] - time_history[i-1]
        weighted_queue += queue_history[i-1] * dt
        total_time += dt
    avg_queue = weighted_queue / total_time if total_time > 0 else 0
    return avg_queue

def main():
    # Параметры варианта (например, n=17 => k=1, m=3, r=0, l=0)
    # Замените на соответствующие вашему варианту
    n = 17
    k = 1  # n mod 4 = 17 mod 4 = 1
    m = 3  # n mod 7 = 17 mod 7 = 3
    r = 0  # (n + k) mod 3 = (17 + 1) mod 3 = 18 mod 3 = 0
    l = 0  # (n + m) mod 5 = (17 + 3) mod 5 = 20 mod 5 = 0

    print(f"Параметры варианта:\nk={k}, m={m}, r={r}, l={l}\n")

    # 1. Теоретическое стационарное распределение
    pi_theor = compute_stationary_distribution(k, m, r, l)
    print("Теоретическое стационарное распределение числа работающих блоков:")
    for i, pi in enumerate(pi_theor):
        print(f"Состояние {i}: {pi:.6f}")
    
    # 2. Средняя длина очереди
    # Теоретически, средняя длина очереди может быть вычислена как сумма (i - (k+2)) * pi_i для i > (k+2)
    avg_queue_theor = 0.0
    for i in range(k + 3, len(pi_theor)):
        avg_queue_theor += (i - (k + 2)) * pi_theor[i]
    print(f"\nТеоретическая средняя длина очереди: {avg_queue_theor:.6f}")

    # 3. Моделирование процесса на 1000 часов
    state_history, queue_history, time_history = simulate_failure_repair(k, m, r, l, T=1000, random_seed=42)
    print("\nИмитация завершена.")

    # 4. Вычисление эмпирического стационарного распределения
    max_state = m + k + 2
    counts = np.zeros(max_state + 1)
    for state in state_history:
        if 0 <= state <= max_state:
            counts[state] += 1
    pi_empirical = counts / len(state_history)

    print("\nЭмпирическое стационарное распределение числа работающих блоков:")
    for i, pi in enumerate(pi_empirical):
        print(f"Состояние {i}: {pi:.6f}")

    # 5. Средняя длина очереди эмпирически
    avg_queue_empirical = compute_average_queue(queue_history, time_history)
    print(f"\nЭмпирическая средняя длина очереди: {avg_queue_empirical:.6f}")

    # 6. Сравнение теоретических и эмпирических значений
    print("\nСравнение теоретических и эмпирических значений:")
    print("{:<10} {:<15} {:<15} {:<15}".format('Состояние', 'Теор. pi', 'Эмпир. pi', 'Погрешность (%)'))
    for i in range(max_state + 1):
        pi_t = pi_theor[i]
        pi_e = pi_empirical[i]
        delta = abs(pi_e - pi_t) / pi_t * 100 if pi_t != 0 else 0
        print(f"{i:<10} {pi_t:<15.6f} {pi_e:<15.6f} {delta:<15.2f}")

    # Средняя длина очереди
    delta_queue = abs(avg_queue_empirical - avg_queue_theor) / avg_queue_theor * 100 if avg_queue_theor != 0 else 0
    print(f"\nСредняя длина очереди:\nТеоретическая: {avg_queue_theor:.6f}\nЭмпирическая: {avg_queue_empirical:.6f}\nПогрешность: {delta_queue:.2f}%")

    # Сохранение результатов в файл для отчёта
    np.savetxt("task3_stationary_distribution_theoretical.csv", pi_theor, delimiter=',', header="Theoretical Pi", comments='')
    np.savetxt("task3_stationary_distribution_empirical.csv", pi_empirical, delimiter=',', header="Empirical Pi", comments='')
    with open("task3_average_queue.txt", "w") as f:
        f.write(f"Theoretical Average Queue: {avg_queue_theor:.6f}\n")
        f.write(f"Empirical Average Queue: {avg_queue_empirical:.6f}\n")
        f.write(f"Relative Error (%): {delta_queue:.2f}\n")

    # Визуализация
    states = np.arange(max_state + 1)
    plt.figure(figsize=(10,6))
    plt.bar(states - 0.2, pi_theor, width=0.4, label='Теоретическое', align='center')
    plt.bar(states + 0.2, pi_empirical, width=0.4, label='Эмпирическое', align='center')
    plt.xlabel('Состояние (число работающих блоков)')
    plt.ylabel('Вероятность')
    plt.title('Сравнение теоретического и эмпирического стационарного распределения')
    plt.legend()
    plt.savefig("task3_stationary_distribution_comparison.png")
    plt.show()

    # График средней длины очереди
    plt.figure(figsize=(8,5))
    plt.plot(time_history, queue_history, label='Длина очереди')
    plt.xlabel('Время (часы)')
    plt.ylabel('Длина очереди')
    plt.title('Динамика длины очереди во времени')
    plt.legend()
    plt.grid(True)
    plt.savefig("task3_queue_dynamics.png")
    plt.show()

if __name__ == "__main__":
    main()
