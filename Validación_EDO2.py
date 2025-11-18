import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definición de Funciones ---

def F_sistema_segundo_orden(x, Y):
    """
    Función del lado derecho del sistema de EDOs de primer orden derivado de:
    y'' + y = sin(x)  =>  y'' = sin(x) - y
    
    Variables: Y[0] = y1 (posición), Y[1] = y2 (velocidad)
    Sistema:
    y1' = y2
    y2' = sin(x) - y1
    
    Parámetros:
    x: Variable independiente (tiempo/posición).
    Y: Vector de soluciones [y1, y2].
    
    Retorna:
    Vector de derivadas [y1', y2'].
    """
    y1 = Y[0]
    # y2 = Y[1] # y1' = y2
    
    dy1dx = Y[1]            # y1' = y2
    dy2dx = np.sin(x) - y1  # y2' = sin(x) - y1
    
    return np.array([dy1dx, dy2dx])

def y_analitica(x):
    """
    Solución analítica para y(x) = 0.5 * sin(x) - 0.5 * x * cos(x)
    """
    return 0.5 * np.sin(x) - 0.5 * x * np.cos(x)

# --- 2. RK4 Solver (Reutilizado del Script Anterior) ---

def RK4_solver_sistema(F, t0, Y0, h, t_final):
    """
    Implementación del método Runge-Kutta de orden 4 para un sistema de EDOs.
    (La variable 't' representa 'x' en este problema)
    """
    T = [t0]
    Y_sol = [Y0]

    t_n = t0
    Y_n = np.array(Y0) 

    while t_n < t_final:
        # Ajuste del último paso
        h_step = min(h, t_final - t_n)
        if h_step == 0:
            break

        # Coeficientes de Runge-Kutta (Vectoriales)
        k1 = F(t_n, Y_n)
        k2 = F(t_n + h_step/2.0, Y_n + h_step/2.0 * k1)
        k3 = F(t_n + h_step/2.0, Y_n + h_step/2.0 * k2)
        k4 = F(t_n + h_step, Y_n + h_step * k3)

        # Nueva aproximación
        Y_n = Y_n + h_step/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t_n = t_n + h_step

        T.append(t_n)
        Y_sol.append(Y_n)

    T_arr = np.array(T)
    Y_sol_arr = np.array(Y_sol)
    
    # y1_num es la primera columna (la solución y(x) original)
    return T_arr, Y_sol_arr[:, 0], Y_sol_arr[:, 1]

# --- 3. Parámetros y Ejecución ---

# Parámetros de la simulación
x0 = 0.0          # Condición inicial x
Y0 = [0.0, 0.0]   # Condiciones iniciales: y1(0)=0, y2(0)=0
x_final = 20.0    # Final del intervalo
h = 0.1           # Tamaño del paso

print(f"⚙️ Iniciando Solución RK4 para EDO de 2do Orden con h = {h} en x=[0, {x_final}]...")

# Ejecución del solver
X_num, Y1_num, Y2_num = RK4_solver_sistema(F_sistema_segundo_orden, x0, Y0, h, x_final)

# Cálculo de la solución analítica para Y1 (y(x))
Y1_analitico = y_analitica(X_num)

print("✅ Solución numérica calculada.")
print("-" * 50)

# --- 4. Cálculo del Error Global Máximo ---

# Error absoluto en cada punto
Error_Abs_Y1 = np.abs(Y1_analitico - Y1_num)

# Error global máximo
Error_Max_Y1 = np.max(Error_Abs_Y1)

print(f"📈 Error Global Máximo (|y_analítica - y1_numérica|): {Error_Max_Y1:.8e}")
print("-" * 50)

# --- 5. Visualización de Resultados (Gráfica) ---

plt.figure(figsize=(10, 6))

# Gráfica de la solución analítica
plt.plot(X_num, Y1_analitico, label='Solución Analítica $y(x)$', color='blue', linewidth=3, alpha=0.6)

# Gráfica de la solución numérica
plt.plot(X_num, Y1_num, 'r--', markersize=2, label=f'Solución Numérica (RK4, $h={h}$)', alpha=0.9)

plt.title('Respuesta del Sistema Subamortiguado con Resonancia')
plt.xlabel('Variable independiente $x$')
plt.ylabel('$y(x) \quad (\mathrm{y}1)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.text(X_num[-1], Y1_analitico[-1], 
         f"Error Máximo: {Error_Max_Y1:.2e}", 
         verticalalignment='bottom', horizontalalignment='right', 
         bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
plt.show()
