import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

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
    if u <= 0.25:  # F(x) = x^2
        return np.sqrt(u)
    else:          # F(x) = 1.5*x - 0.5
        return (u + 0.5) / 1.5

def generate_sample(n=5000, random_seed=42):
    np.random.seed(random_seed)
    U = np.random.rand(n)  # U ~ Uniform(0,1)
    X = np.array([F_inv(u) for u in U])
    return X

def plot_histogram(X, bins='auto'):
    """
    Построение гистограммы выборки
    """
    plt.hist(X, bins=bins, density=True, alpha=0.6, edgecolor='black')
    plt.title("Гистограмма выборки")
    plt.xlabel("Значение")
    plt.ylabel("Относительная частота")
    plt.grid(True)
    plt.show()

def random_walk_trajectories(X, num_trajectories=3, steps_per_trajectory=50):
    """
    Рассматриваем элементы выборки X как шаги случайного блуждания.
    Для наглядности построим несколько траекторий (num_trajectories).
    Каждой траектории берём steps_per_trajectory подряд и суммируем.
    """
    plt.figure(figsize=(8, 5))
    
    index = 0
    for i in range(num_trajectories):
        # берём отрезок X[index : index+steps_per_trajectory]
        steps = X[index : index + steps_per_trajectory]
        index += steps_per_trajectory
        
        # случайное блуждание - кумулятивная сумма
        rw = np.cumsum(steps)
        plt.plot(rw, label=f"Траектория {i+1}")
        
    plt.title("Случайные блуждания")
    plt.xlabel("Номер шага")
    plt.ylabel("Положение блуждания")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_theoretical_moments():
    """
    Теоретические M(X) и D(X).
    f(x) = 2x  при x в [0, 0.5],   => интеграл(0..0.5) ...
           1.5 при x в (0.5..1].
    Для наглядности сразу считаем аналитически:
      E(X) ~ 0.6458333
      Var(X) ~ 0.0516
    """
    E = 31/48            # 0.6458333...
    # E(X^2) = 15/32 = 0.46875
    # => Var(X) = E(X^2) - [E(X)]^2 = ~ 0.0516
    Var = (15/32) - (31/48)**2
    return E, Var

def sample_moments_for_subsamples(X, step=200, max_k=25):
    """
    Вычисляем выборочные моменты (среднее и дисперсию)
    для объемов n = 200*k, k = 1..25.
    Считаем относительную погрешность (delta(n)) и строим графики.
    """
    E_th, Var_th = compute_theoretical_moments()
    ns = []
    mean_errors = []
    var_errors = []
    
    for k in range(1, max_k+1):
        n = step * k
        sample_subset = X[:n]
        mean_ = np.mean(sample_subset)
        var_ = np.var(sample_subset, ddof=1)  # несмещенная дисперсия
        
        mean_error = abs(mean_ - E_th) / abs(E_th)
        var_error  = abs(var_ - Var_th) / abs(Var_th)
        
        ns.append(n)
        mean_errors.append(mean_error)
        var_errors.append(var_error)
    
    # График зависимости относительной погрешности
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ns, mean_errors, marker='o')
    plt.title("Относительная погрешность для E(X)")
    plt.xlabel("n")
    plt.ylabel("Погрешность")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(ns, var_errors, marker='o', color='red')
    plt.title("Относительная погрешность для Var(X)")
    plt.xlabel("n")
    plt.ylabel("Погрешность")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def chi_square_test(X, bins=10):
    """
    Проверка согласия распределения выборки с теоретическим F(x)
    с помощью критерия хи-квадрат.
    
    Для корректной работы нужно:
      - Разбить (0,1) на интервалы (bins),
      - Для каждого интервала посчитать наблюдённые частоты,
      - Вычислить теоретические вероятности (p_i) на каждом интервале
        из F(x).
      - Посчитать статистику хи-квадрат и сравнить с критическим значением.
    Здесь приведён упрощённый набросок.
    """
    # Разбиваем [0,1] на интервалы:
    edges = np.linspace(0, 1, bins+1)
    observed, _ = np.histogram(X, bins=edges)
    n = len(X)
    
    # Теоретическая функция распределения
    def F(x):
        if x < 0: 
            return 0
        elif x <= 0.5: 
            return x*x
        elif x <= 1: 
            return 1.5*x - 0.5
        else:
            return 1
    
    # Теоретические вероятности для каждого интервала
    p = []
    for i in range(bins):
        left = edges[i]
        right = edges[i+1]
        p_i = F(right) - F(left)
        p.append(p_i)
    p = np.array(p)
    
    # Проверяем, чтобы все p_i > 0
    # (иначе требуется объединение интервалов с малой вероятностью)
    expected = n * p
    
    chi2_stat = np.sum((observed - expected)**2 / expected)
    
    # Степени свободы: bins - 1 - количество оценённых параметров
    # Параметров тут фактически 0 (мы всё знаем теоретически), но на практике иногда берут поправку.
    df = bins - 1
    
    # Можно вывести итог:
    print("Chi-square stat =", chi2_stat, ", df =", df)
    # Для более точного вывода — сравнить с квантилем распределения хи-квадрат,
    # но это уже деталь (можно воспользоваться scipy.stats)
    
def main():
    # 1) Генерация выборки
    X = generate_sample(n=5000, random_seed=42)
    
    # 2) Гистограмма
    plot_histogram(X, bins='auto')
    
    # 3) Случайное блуждание (3-5 траекторий)
    random_walk_trajectories(X, num_trajectories=3, steps_per_trajectory=100)
    
    # 4) Вычисление теоретических и выборочных моментов
    E_th, Var_th = compute_theoretical_moments()
    print(f"Теоретическое мат. ожидание: {E_th:.5f}, дисперсия: {Var_th:.5f}")
    print(f"Выборочное среднее: {np.mean(X):.5f}, выборочная дисперсия: {np.var(X, ddof=1):.5f}")
    
    # Построить графики зависимости относительной погрешности
    sample_moments_for_subsamples(X, step=200, max_k=25)
    
    # 5) Критерий хи-квадрат
    chi_square_test(X, bins=10)

if __name__ == "__main__":
    main()
