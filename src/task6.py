import numpy as np
import matplotlib.pyplot as plt

def simulate_wiener_process(T=10.0, N=1000, random_seed=42):
    """
    Моделирование винеровского процесса W_t на [0, T].
    Используем дискретизацию: шаг по времени = T/N,
    приращения ~ Normal(0, sqrt(dt)).
    """
    np.random.seed(random_seed)
    dt = T / N
    increments = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(increments)
    W = np.insert(W, 0, 0)  # W(0)=0
    times = np.linspace(0, T, N+1)
    return times, W

def main():
    # 1) Смоделируем несколько траекторий
    T = 10
    N = 10000
    plt.figure(figsize=(10, 5))
    
    for i in range(3):
        t, W = simulate_wiener_process(T, N, random_seed=42+i)
        plt.plot(t, W, label=f"Траектория {i+1}")
    
    plt.legend()
    plt.title("Несколько траекторий винеровского процесса")
    plt.xlabel("t")
    plt.ylabel("W(t)")
    plt.grid(True)
    plt.show()
    
    # 2) Построим графики y = ±(1 ± eps)*sqrt(2t ln ln t) при малом eps
    #    (осторожно, ln ln t не определён при t<=1, требуется t>1, плюс поведение при t большой)
    
    # 3) Проверка распределения M_t = sup_{0<=s<=t} W_s и W_t
    
    # и т.д.

if __name__ == "__main__":
    main()
