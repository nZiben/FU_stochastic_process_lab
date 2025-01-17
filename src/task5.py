import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

def simulate_srw(n_steps=1000, random_seed=42):
    """
    Простое симметричное случайное блуждание S_k:
    S_0=0, S_{k+1} = S_k + Xi_k,
    где Xi_k ~ +/-1 с вероятностью 1/2.
    """
    np.random.seed(random_seed)
    xi = np.random.choice([-1, 1], size=n_steps)
    S = np.concatenate(([0], np.cumsum(xi)))
    return S

def main():
    # 1) Моделирование простого симметричного СБ на 2n=1000 шагов
    n = 500  # тогда 2n=1000
    S = simulate_srw(n_steps=2*n, random_seed=42)
    
    # 2) Рисуем несколько траекторий
    plt.figure(figsize=(8, 5))
    for i in range(3):
        S_ = simulate_srw(n_steps=2*n, random_seed=42+i)
        plt.plot(S_, label=f"Блуждание {i+1}")
    plt.legend()
    plt.title("Несколько траекторий случайного блуждания")
    plt.xlabel("Шаг")
    plt.ylabel("Положение")
    plt.grid(True)
    plt.show()
    
    # 3) Проверка P(0){S1!=0, ..., S_{2n}!=0} = p_{00}(2n)
    #    Требует аккуратной оценки. p_{00}(2n) - вероятность вернуться в 0 за 2n шагов.
    #    p_{00}(2n) = C(2n, n)/(2^(2n)) (известная формула для симм. СБ).
    
    # 4) Выборка моментов времени tau (последнее нахождение в 0), mu (время в полож. полуплоскости), M_n (максимум)
    #    и проверка арксинус-распределения.
    
    # и т.д.
    # Приведён лишь каркас.

if __name__ == "__main__":
    main()
