import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Аналітична функція для варіанта 5 (n = 5)
def f_exact(x, n=5):
    return n * np.sin(np.pi * n * x)

# Помножена на 10 функція
def f_exact_x10(x, n=5):
    return 10 * f_exact(x, n)

# Інтервал [0, pi]
A, B = 0.0, np.pi
T = B - A
w0 = 2.0 * np.pi / T

# 2. Обчислення коефіцієнтів a_k та b_k ряду Фур'є
def compute_fourier_coefficients(N, n=5, scale=1.0):
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    
    def func(x):
        return scale * f_exact(x, n)
        
    integral_a0, _ = quad(func, A, B)
    a[0] = (2.0 / T) * integral_a0
    
    for k in range(1, N + 1):
        integrand_a, _ = quad(lambda x: func(x) * np.cos(k * w0 * x), A, B)
        a[k] = (2.0 / T) * integrand_a
        
        integrand_b, _ = quad(lambda x: func(x) * np.sin(k * w0 * x), A, B)
        b[k] = (2.0 / T) * integrand_b
        
    return a, b

# 3. Наближення рядом Фур'є порядку N
def fourier_approximation(x, a, b, N):
    result = a[0] / 2.0
    for k in range(1, N + 1):
        result += a[k] * np.cos(k * w0 * x) + b[k] * np.sin(k * w0 * x)
    return result

# 4. Побудова графіків в одному вікні
def plot_fourier_analysis(a, b, N=10, title_prefix="", filename="fourier_plot.png", is_x10=False):
    x_vals = np.linspace(A, B, 1000)
    y_exact = f_exact_x10(x_vals) if is_x10 else f_exact(x_vals)
    y_approx = fourier_approximation(x_vals, a, b, N)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title(f"Лабораторна робота №1 — {title_prefix}")
    
    # Порівняння точної функції та її наближення
    axs[0, 0].plot(x_vals, y_exact, 'b-', label='Точна $f(x)$', linewidth=1.5)
    axs[0, 0].plot(x_vals, y_approx, 'r--', label=f'Ряд Фур\'є ($N={N}$)', linewidth=1.2)
    axs[0, 0].set_title(f'Порівняння ({title_prefix})')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Графіки гармонік (k = 1..10)
    for k in range(1, N + 1):
        harmonic = a[k] * np.cos(k * w0 * x_vals) + b[k] * np.sin(k * w0 * x_vals)
        axs[0, 1].plot(x_vals, harmonic, label=f'k={k}')
    axs[0, 1].set_title(f'Гармоніки k=1..{N} ({title_prefix})')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].grid(True)
    axs[0, 1].legend(ncol=2, fontsize='small')
    
    # Амплітудний спектр C_k
    k_indices = np.arange(0, N + 1)
    C_k = np.sqrt(a**2 + b**2)
    C_k[0] = abs(a[0]) / 2.0
    axs[1, 0].stem(k_indices, C_k, basefmt=" ")
    axs[1, 0].set_title(f'Амплітудний спектр $C_k$ ({title_prefix})')
    axs[1, 0].set_xlabel('k')
    axs[1, 0].set_ylabel('Амплітуда')
    axs[1, 0].grid(True)
    
    # Коефіцієнти a_k та b_k у частотній області
    width = 0.35
    axs[1, 1].bar(k_indices - width/2, a, width, label='$a_k$', color='orange')
    axs[1, 1].bar(k_indices + width/2, b, width, label='$b_k$', color='green')
    axs[1, 1].set_title(f'Коефіцієнти $a_k$ та $b_k$ ({title_prefix})')
    axs[1, 1].set_xlabel('k')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# 5. Оцінка відносної похибки наближення
def compute_relative_error(a, b, N, num_points=1000, is_x10=False):
    x_vals = np.linspace(A, B, num_points)
    y_exact = f_exact_x10(x_vals) if is_x10 else f_exact(x_vals)
    y_approx = fourier_approximation(x_vals, a, b, N)
    
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    if trapz_func is None:
        from scipy.integrate import trapezoid
        trapz_func = trapezoid

    error_l2 = np.sqrt(trapz_func((y_exact - y_approx) ** 2, x_vals))
    norm_exact = np.sqrt(trapz_func(y_exact ** 2, x_vals))
    
    return (error_l2 / norm_exact) * 100.0

# 6. Зберігання результатів у файл
def save_results_to_file(filename, N, a, b, rel_error, title="Основна функція"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Результати: {title} (N = {N})\n")
        f.write("Обчислені коефіцієнти тригонометричного ряду Фур'є:\n")
        f.write(f"a_0 / 2 = {a[0]/2:.6f}\n")
        for k in range(1, N + 1):
            f.write(f"k={k:2d}: a_k = {a[k]:10.6f}, b_k = {b[k]:10.6f}\n")
        f.write(f"Відносна похибка наближення: {rel_error:.4f}%\n")

# 7. Головна програма
def main():
    N = 10
    
    # 1. Основна функція f(x) = 5*sin(5*pi*x)
    a, b = compute_fourier_coefficients(N, scale=1.0)
    rel_error = compute_relative_error(a, b, N, is_x10=False)
    
    print(f"Основна функція f(x):")
    print(f"Відносна похибка (N = {N}): {rel_error:.2f}%\n")
    
    save_results_to_file("fourier_results.txt", N, a, b, rel_error, title="Основна функція")
    plot_fourier_analysis(a, b, N, title_prefix="Основна функція", filename="fourier_analysis_var5.png", is_x10=False)
    
    # 2. Помножена функція 10 * f(x) = 50*sin(5*pi*x)
    a_x10, b_x10 = compute_fourier_coefficients(N, scale=10.0)
    rel_error_x10 = compute_relative_error(a_x10, b_x10, N, is_x10=True)
    
    print(f"Помножена функція 10 * f(x):")
    print(f"Відносна похибка (N = {N}): {rel_error_x10:.2f}%")
    
    save_results_to_file("fourier_results_x10.txt", N, a_x10, b_x10, rel_error_x10, title="Функція * 10")
    plot_fourier_analysis(a_x10, b_x10, N, title_prefix="Функція * 10", filename="fourier_analysis_var5_x10.png", is_x10=True)

if __name__ == "__main__":
    main()
