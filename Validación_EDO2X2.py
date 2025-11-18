import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definición de Funciones ---

def F_sistema(t, Y):
    """
    Función del lado derecho del sistema de EDOs.
    
    Parámetros:
    t: Variable independiente (tiempo).
    Y: Vector de soluciones [x(t), y(t)].
    
    Retorna:
    Vector de derivadas [x'(t), y'(t)].
    """
    x = Y[0]
    y = Y[1]
    
    # Sistema:
    # x'(t) = 3*x + 4*y
    # y'(t) = -4*x + 3*y
    dxdt = 3.0 * x + 4.0 * y
    dydt = -4.0 * x + 3.0 * y
    
    return np.array([dxdt, dydt])

def x_analitica(t):
    """Solución analítica para x(t)."""
    return np.exp(3.0 * t) * np.cos(4.0 * t)

def y_analitica(t):
    """Solución analítica para y(t)."""
    return -np.exp(3.0 * t) * np.sin(4.0 * t)

# --- 2. Adaptación del Solver RK4 para Sistemas ---

def RK4_solver_sistema(F, t0, Y0, h, t_final):
    """
    Implementa el método Runge-Kutta de orden 4 para un sistema de EDOs.
    
    F: Función F(t, Y) que retorna un vector de derivadas [x', y', ...].
    Y0: Vector de condiciones iniciales [x0, y0, ...].
    """
    # Inicialización de las listas de resultados
    T = [t0]
    Y_sol = [Y0]

    t_n = t0
    Y_n = np.array(Y0) # Aseguramos que es un array numpy para operaciones vectoriales

    while t_n < t_final:
        # Ajuste del último paso
        if t_n + h > t_final:
            h_step = t_final - t_n
            if h_step == 0:
                break
        else:
            h_step = h

        # Coeficientes de Runge-Kutta (Ahora son vectores)
        k1 = F_sistema(t_n, Y_n)
        k2 = F_sistema(t_n + h_step/2.0, Y_n + h_step/2.0 * k1)
        k3 = F_sistema(t_n + h_step/2.0, Y_n + h_step/2.0 * k2)
        k4 = F_sistema(t_n + h_step, Y_n + h_step * k3)

        # Cálculo del nuevo vector de aproximación
        Y_n_mas_1 = Y_n + h_step/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # Actualización de valores
        t_n = t_n + h_step
        Y_n = Y_n_mas_1

        # Almacenamiento de resultados
        T.append(t_n)
        Y_sol.append(Y_n)

    # Convertir a arrays numpy para fácil manipulación
    T_arr = np.array(T)
    Y_sol_arr = np.array(Y_sol)
    
    # Separar las soluciones: X_numerico es la primera columna, Y_numerico la segunda
    X_num = Y_sol_arr[:, 0]
    Y_num = Y_sol_arr[:, 1]
    
    return T_arr, X_num, Y_num

# --- 3. Parámetros y Ejecución ---

# Parámetros de la simulación
t0 = 0.0          # Tiempo inicial
Y0 = [1.0, 0.0]   # Condición inicial: x(0)=1, y(0)=0
t_final = 5.0     # Final del intervalo
h = 0.1           # Tamaño del paso

print(f"⚙️ Iniciando Solución RK4 para el Sistema con h = {h} en t=[0, {t_final}]...")

# Ejecución del solver
T_num, X_num, Y_num = RK4_solver_sistema(F_sistema, t0, Y0, h, t_final)

# Cálculo de las soluciones analíticas
X_analitico = x_analitica(T_num)
Y_analitico = y_analitica(T_num)

print("✅ Soluciones numéricas calculadas.")
print("-" * 50)

# --- 4. Cálculo del Error Global Máximo ---

# Error absoluto en cada punto
Error_Abs_X = np.abs(X_analitico - X_num)
Error_Abs_Y = np.abs(Y_analitico - Y_num)

# Error global máximo
Error_Max_X = np.max(Error_Abs_X)
Error_Max_Y = np.max(Error_Abs_Y)

print(f"📈 Error Global Máximo para X(t): {Error_Max_X:.8e}")
print(f"📈 Error Global Máximo para Y(t): {Error_Max_Y:.8e}")
print("-" * 50)


# --- 5. Visualización de Resultados (Gráficas) ---

## 5a) Comparación de x_numerico vs x_analitico
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)

plt.plot(T_num, X_analitico, label='Analítica: $x(t) = e^{3t} \cos(4t)$', color='blue', linewidth=2)
plt.plot(T_num, X_num, 'r--', markersize=3, label=f'Numérica (RK4, $h={h}$)', alpha=0.7)

plt.title('Comparación de $x(t)$ (RK4 vs Analítica)')
plt.xlabel('Tiempo $t$')
plt.ylabel('$x(t)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

## 5b) Comparación de y_numerico vs y_analitico
plt.subplot(1, 2, 2)

plt.plot(T_num, Y_analitico, label='Analítica: $y(t) = -e^{3t} \sin(4t)$', color='green', linewidth=2)
plt.plot(T_num, Y_num, 'm--', markersize=3, label=f'Numérica (RK4, $h={h}$)', alpha=0.7)

plt.title('Comparación de $y(t)$ (RK4 vs Analítica)')
plt.xlabel('Tiempo $t$')
plt.ylabel('$y(t)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# --- Gráfico de Error (Opcional pero informativo) ---

plt.figure(figsize=(12, 4))
plt.plot(T_num, Error_Abs_X, label='Error Absoluto $|x_{analitico} - x_{RK4}|$', color='darkorange')
plt.plot(T_num, Error_Abs_Y, label='Error Absoluto $|y_{analitico} - y_{RK4}|$', color='purple')
plt.title('Errores Absolutos para $x(t)$ e $y(t)$ en función del tiempo')
plt.xlabel('Tiempo $t$')
plt.ylabel('Error Absoluto')
plt.legend()
plt.yscale('log') # Escala logarítmica para ver la magnitud
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
