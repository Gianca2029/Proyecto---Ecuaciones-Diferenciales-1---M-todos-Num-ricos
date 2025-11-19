import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Definición del Sistema y Parámetros
# ==========================================

def F(t, Y, params):
    """
    Define las ecuaciones del sistema Lotka-Volterra.
    
    Args:
        t (float): Tiempo actual.
        Y (array): Vector de estado [x, y] (Presas, Depredadores).
        params (list): [alpha, beta, delta, gamma].
        
    Returns:
        numpy.array: Derivadas [dx/dt, dy/dt].
    """
    x, y = Y
    alpha, beta, delta, gamma = params
    
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    
    return np.array([dxdt, dydt])

# Parámetros del problema
alpha = 1.1  # Tasa de crecimiento de presas
beta  = 0.4  # Tasa de depredación
delta = 0.1  # Eficiencia de conversión
gamma = 0.4  # Tasa de mortalidad de depredadores

params = [alpha, beta, delta, gamma]

# Condiciones iniciales y tiempo
Y0 = np.array([10.0, 5.0]) # x(0)=10, y(0)=5
t_start = 0.0
t_end = 50.0
h = 0.01

# ==========================================
# 2. Implementación de Métodos Numéricos
# ==========================================

def heun_step(t, Y, h, func, params):
    """
    Realiza un paso de integración usando el método de Heun (RK2).
    """
    k1 = func(t, Y, params)
    k2 = func(t + h, Y + h * k1, params)
    return Y + (h / 2.0) * (k1 + k2)

def rk4_step(t, Y, h, func, params):
    """
    Realiza un paso de integración usando Runge-Kutta orden 4 (RK4).
    """
    k1 = func(t, Y, params)
    k2 = func(t + 0.5 * h, Y + 0.5 * h * k1, params)
    k3 = func(t + 0.5 * h, Y + 0.5 * h * k2, params)
    k4 = func(t + h, Y + h * k3, params)
    
    return Y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def resolver_sistema(metodo_step, t0, t_end, h, Y0, func, params):
    """
    Bucle principal de simulación temporal.
    """
    times = np.arange(t0, t_end + h, h)
    num_steps = len(times)
    
    # Inicializar array de resultados (filas=tiempo, columnas=variables)
    Y_history = np.zeros((num_steps, len(Y0)))
    Y_history[0] = Y0
    
    curr_Y = Y0.copy()
    
    for i in range(num_steps - 1):
        t = times[i]
        curr_Y = metodo_step(t, curr_Y, h, func, params)
        Y_history[i+1] = curr_Y
        
    return times, Y_history

# ==========================================
# 3. Ejecución y Comparación
# ==========================================

print("Iniciando simulación...")

# Ejecución RK2 (Heun)
t_rk2, res_rk2 = resolver_sistema(heun_step, t_start, t_end, h, Y0, F, params)
x_rk2, y_rk2 = res_rk2[:, 0], res_rk2[:, 1]

# Ejecución RK4
t_rk4, res_rk4 = resolver_sistema(rk4_step, t_start, t_end, h, Y0, F, params)
x_rk4, y_rk4 = res_rk4[:, 0], res_rk4[:, 1]

# Cálculo de diferencias
diff_x = np.abs(x_rk2 - x_rk4)
diff_y = np.abs(y_rk2 - y_rk4)
max_diff_x = np.max(diff_x)
max_diff_y = np.max(diff_y)

print(f"Diferencia máxima absoluta en Presas (x): {max_diff_x:.6f}")
print(f"Diferencia máxima absoluta en Depredadores (y): {max_diff_y:.6f}")

# ==========================================
# 4. Presentación de Resultados (Tabla)
# ==========================================

puntos_interes = [0, 10, 20, 30, 40, 50]

print("\n### Tabla Comparativa de Resultados")
print("| Tiempo (t) | x (RK2) | y (RK2) | x (RK4) | y (RK4) | Diff x | Diff y |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

for t_val in puntos_interes:
    # Encontrar el índice más cercano al tiempo deseado
    idx = np.abs(t_rk4 - t_val).argmin()
    
    print(f"| {t_rk4[idx]:.1f} | {x_rk2[idx]:.4f} | {y_rk2[idx]:.4f} | "
          f"{x_rk4[idx]:.4f} | {y_rk4[idx]:.4f} | "
          f"{diff_x[idx]:.2e} | {diff_y[idx]:.2e} |")

# ==========================================
# 5. Visualización
# ==========================================

fig = plt.figure(figsize=(14, 6))

# Gráfico 1: Series de Tiempo
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(t_rk2, x_rk2, 'b--', label='Presas (RK2)', alpha=0.7)
ax1.plot(t_rk2, y_rk2, 'r--', label='Depredadores (RK2)', alpha=0.7)
# RK4 se dibuja con línea sólida encima
ax1.plot(t_rk4, x_rk4, 'b-', label='Presas (RK4)', linewidth=1)
ax1.plot(t_rk4, y_rk4, 'r-', label='Depredadores (RK4)', linewidth=1)

ax1.set_title('Dinámica Poblacional: RK2 vs RK4')
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Población')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Espacio Fase
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x_rk2, y_rk2, 'g--', label='Ciclo RK2', alpha=0.6)
ax2.plot(x_rk4, y_rk4, 'k-', label='Ciclo RK4', linewidth=1)

ax2.set_title('Espacio Fase (Presas vs Depredadores)')
ax2.set_xlabel('Presas (x)')
ax2.set_ylabel('Depredadores (y)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# --- CAMBIO: Guardar la figura antes de mostrarla ---
archivo_salida = 'lotka_volterra_comparacion.png'
plt.savefig(archivo_salida, dpi=300)
print(f"\n[INFO] La gráfica se ha guardado como '{archivo_salida}' en la carpeta actual.")

# Intentar mostrar la ventana
try:
    plt.show()
except Exception as e:
    print(f"No se pudo abrir la ventana gráfica interactiva: {e}")
    print("Por favor revisa el archivo PNG generado.")
