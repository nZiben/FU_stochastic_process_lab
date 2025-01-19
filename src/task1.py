import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import os
from scipy.stats import chisquare  # Исправленный импорт

def F_inv(u):
    """
    Обратная функция распределения (метод инверсии).
    F(x) = {
      x^2,                      x in [0, 0.5],
      1.5*x - 0.5,             x in (0.5, 1].
    }
    F_inv(u) = ?
    
    Для 0 <= u <= 0.25:  X = sqrt(u),  т.к. F(x)=x^2
    Для 0.25 < u <= 1:   X = (u + 0.5)/1.5
    """
    return np.where(u <= 0.25, np.sqrt(u), (u + 0.5) / 1.5)

def generate_sample(n=5000, random_seed=42):
    """
    Генерация выборки из n случайных величин с заданным распределением.
    
    Параметры:
    - n: размер выборки
    - random_seed: начальное значение для генератора случайных чисел
    
    Возвращает:
    - X: массив сгенерированных случайных величин
    """
    np.random.seed(random_seed)
    U = np.random.rand(n)  # U ~ Uniform(0,1)
    X = F_inv(U)
    return X

def plot_histogram(X, filepath, bins='auto'):
    """
    Построение и сохранение гистограммы выборки.
    
    Параметры:
    - X: массив случайных величин
    - filepath: путь для сохранения графика
    - bins: количество бинов или метод их определения
    """
    plt.figure(figsize=(10,6))
    plt.hist(X, bins=bins, density=True, alpha=0.6, edgecolor='black', color='skyblue', label='Эмпирическое распределение')
    
    # Теоретическая функция плотности
    x_vals = np.linspace(0, 1, 400)
    f_x = np.where(x_vals <= 0.5, 2*x_vals, 1.5*np.ones_like(x_vals))
    plt.plot(x_vals, f_x, 'r-', label='Теоретическая плотность')
    
    plt.title("Гистограмма выборки")
    plt.xlabel("Значение")
    plt.ylabel("Относительная частота")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    print(f"Гистограмма сохранена в {filepath}")

def random_walk_trajectories(X, filepath, num_trajectories=3, steps_per_trajectory=100):
    """
    Генерация и сохранение графика траекторий симметричного случайного блуждания.
    
    Параметры:
    - X: массив случайных величин
    - filepath: путь для сохранения графика
    - num_trajectories: количество траекторий для построения
    - steps_per_trajectory: количество шагов в каждой траектории
    """
    plt.figure(figsize=(12, 8))
    
    index = 0
    for i in range(num_trajectories):
        # Берём отрезок X[index : index+steps_per_trajectory]
        steps = X[index : index + steps_per_trajectory]
        index += steps_per_trajectory
        
        # Симметричное случайное блуждание: шаг может быть +1 или -1
        directions = np.where(steps < 0.5, -1, 1)  # Например, если X < 0.5, шаг -1, иначе +1
        rw = np.cumsum(directions)
        plt.plot(rw, label=f"Траектория {i+1}")
        
    plt.title("Симметричные случайные блуждания")
    plt.xlabel("Номер шага")
    plt.ylabel("Положение блуждания")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    print(f"Траектории случайного блуждания сохранены в {filepath}")

def compute_theoretical_moments():
    """
    Вычисление теоретических математического ожидания и дисперсии.
    
    Возвращает:
    - E: математическое ожидание
    - Var: дисперсия
    """
    # Для x в [0,0.5]: f(x) = 2x
    # Для x в (0.5,1]: f(x) = 1.5
    # E(X) = ∫0^0.5 x*2x dx + ∫0.5^1 x*1.5 dx = 2∫0^0.5 x^2 dx + 1.5∫0.5^1 x dx
    E = 2 * ( (0.5**3) / 3 ) + 1.5 * ( (1**2)/2 - (0.5**2)/2 )
    # E(X^2) = ∫0^0.5 x^2*2x dx + ∫0.5^1 x^2*1.5 dx = 2∫0^0.5 x^3 dx + 1.5∫0.5^1 x^2 dx
    E_x2 = 2 * ( (0.5**4) / 4 ) + 1.5 * ( (1**3)/3 - (0.5**3)/3 )
    Var = E_x2 - E**2
    return E, Var

def sample_moments_for_subsamples(X, filepath, step=200, max_k=25):
    """
    Вычисление выборочных моментов и сохранение графиков относительной погрешности.
    
    Параметры:
    - X: массив случайных величин
    - filepath: путь для сохранения графика
    - step: шаг увеличения размера выборки
    - max_k: максимальный множитель шага
    """
    E_th, Var_th = compute_theoretical_moments()
    ns = []
    mean_errors = []
    var_errors = []
    
    for k in range(1, max_k+1):
        n = step * k
        if n > len(X):
            print(f"Пропуск n={n}, так как превышает размер выборки.")
            continue
        sample_subset = X[:n]
        mean_ = np.mean(sample_subset)
        var_ = np.var(sample_subset, ddof=1)  # Несмещённая дисперсия
        
        mean_error = abs(mean_ - E_th) / abs(E_th)
        var_error  = abs(var_ - Var_th) / abs(Var_th)
        
        ns.append(n)
        mean_errors.append(mean_error)
        var_errors.append(var_error)
    
    # Создание директории для графиков, если не существует
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # График относительной погрешности для средних
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ns, mean_errors, marker='o', color='blue')
    plt.title("Относительная погрешность для E(X)")
    plt.xlabel("Размер выборки (n)")
    plt.ylabel("Погрешность")
    plt.grid(True)
    
    # График относительной погрешности для дисперсий
    plt.subplot(1, 2, 2)
    plt.plot(ns, var_errors, marker='o', color='red')
    plt.title("Относительная погрешность для Var(X)")
    plt.xlabel("Размер выборки (n)")
    plt.ylabel("Погрешность")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Графики погрешности моментов сохранены в {filepath}")

def chi_square_test(X, filepath, bins=10):
    """
    Проведение теста согласия \(\chi^2\) между эмпирическим распределением выборки и теоретическим.
    Результаты сохраняются в текстовый файл.
    
    Параметры:
    - X: массив случайных величин
    - filepath: путь для сохранения результатов теста
    - bins: количество интервалов для гистограммы
    """
    # Разбиваем [0,1] на интервалы:
    edges = np.linspace(0, 1, bins+1)
    observed, _ = np.histogram(X, bins=edges)
    
    # Теоретическая функция распределения
    def F(x):
        if x < 0: 
            return 0
        elif x <= 0.5: 
            return x**2
        elif x <= 1: 
            return 1.5*x - 0.5
        else:
            return 1
    
    # Вычисление теоретических вероятностей для каждого интервала
    p = []
    for i in range(bins):
        left = edges[i]
        right = edges[i+1]
        p_i = F(right) - F(left)
        p.append(p_i)
    p = np.array(p)
    
    # Проверка, чтобы все p_i >= 0.05 для надежности \(\chi^2\) теста
    # Если нет, объединим малые бинны с соседними
    min_expected = 5  # Минимальное ожидаемое число событий
    expected = p * len(X)
    
    while True:
        small_bins = expected < min_expected
        if not small_bins.any():
            break
        # Объединяем малые бинны с соседним
        first_small = np.where(small_bins)[0][0]
        if first_small == 0:
            # Объединяем с правым
            p[first_small+1] += p[first_small]
            observed[first_small+1] += observed[first_small]
        else:
            # Объединяем с левым
            p[first_small-1] += p[first_small]
            observed[first_small-1] += observed[first_small]
        # Удаляем объединённый бин
        p = np.delete(p, first_small)
        observed = np.delete(observed, first_small)
        expected = p * len(X)
    
    # Проведение \(\chi^2\) теста
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    
    # Сохранение результатов теста в файл с указанием кодировки utf-8
    with open(filepath, "w", encoding='utf-8') as f:
        f.write("Тест согласия χ²\n")
        f.write(f"Число интервалов после объединения: {len(observed)}\n")
        f.write(f"Статистика χ²: {chi2_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
    
    print(f"Результаты теста χ² сохранены в {filepath}")
    return chi2_stat, p_value

def main():
    # Создание директории для графиков, если не существует
    os.makedirs("charts", exist_ok=True)
    
    # 1) Генерация выборки
    X = generate_sample(n=5000, random_seed=42)
    
    # 2) Построение и сохранение гистограммы
    plot_histogram(X, filepath="charts/task1_histogram.png", bins=30)
    
    # 3) Построение и сохранение траекторий случайного блуждания
    random_walk_trajectories(X, filepath="charts/task1_random_walk_trajectories.png", 
                             num_trajectories=3, steps_per_trajectory=100)
    
    # 4) Вычисление теоретических и выборочных моментов
    E_th, Var_th = compute_theoretical_moments()
    E_sample = np.mean(X)
    Var_sample = np.var(X, ddof=1)
    print(f"Теоретическое математическое ожидание: {E_th:.5f}")
    print(f"Теоретическая дисперсия: {Var_th:.5f}")
    print(f"Выборочное среднее: {E_sample:.5f}")
    print(f"Выборочная дисперсия: {Var_sample:.5f}")
    
    # 5) Построение и сохранение графиков относительной погрешности
    sample_moments_for_subsamples(X, filepath="charts/task1_moments_error.png", 
                                  step=200, max_k=25)
    
    # 6) Проведение теста χ² и сохранение результатов
    chi2_stat, p_value = chi_square_test(X, filepath="charts/task1_chi_square_test.txt", bins=10)
    print(f"\nРезультаты теста χ²:")
    print(f"Статистика χ²: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

if __name__ == "__main__":
    main()