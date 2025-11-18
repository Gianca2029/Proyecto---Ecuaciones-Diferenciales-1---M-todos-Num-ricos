import numpy as np
import pandas as pd # Importamos pandas para la tabla formateada
import matplotlib.pyplot as plt

# --- 1. Definición de Funciones del Sistema y Analíticas ---

def F_sistema(t, Y):
    """
    Función del lado derecho del sistema de EDOs:
    x'(t) = 3*x + 4*y
    y'(t) = -4*x + 3*y
    """
    x = Y[0]
    y = Y[1]
    dxdt = 3.0 * x + 4.0 * y
    dydt = -4.0 * x + 3.0 * y
    return np.array([dxdt, dydt])

def x_analitica(t):
    """Solución analítica para x(t) = exp(3*t) * cos(4*t)"""
    return np.exp(3.0 * t) * np.cos(4.0 * t)

def y_analitica(t):
    """Solución analítica para y(t) = -exp(3*t) * sin(4*t)"""
    return -np.exp(3.0 * t) * np.sin(4.0 * t)

def RK4_solver_sistema(F, t0, Y0, h, t_final):
    """
    Implementación del método Runge-Kutta de orden 4 para un sistema de EDOs.
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
    
    return T_arr, Y_sol_arr[:, 0], Y_sol_arr[:, 1] # Retorna T, X_num, Y_num

# --- 2. Parámetros y Bucle de Análisis ---

# Parámetros fijos
t0 = 0.0          # Tiempo inicial
Y0 = [1.0, 0.0]   # Condiciones iniciales
t_final = 5.0     # Final del intervalo

# 1. Definir lista de tamaños de paso
h_values = [0.1, 0.01, 0.001]
results = []

print("🔬 Iniciando estudio de convergencia para el sistema 2x2...")
print("-" * 70)

# 2. Iterar sobre la lista de h
for h in h_values:
    # Ejecución del solver RK4
    T_num, X_num, Y_num = RK4_solver_sistema(F_sistema, t0, Y0, h, t_final)
    
    # Soluciones analíticas en los mismos puntos T
    X_analitico = x_analitica(T_num)
    Y_analitico = y_analitica(T_num)
    
    # 3. Calcular el Error Máximo Absoluto para X(t) y Y(t)
    Error_Max_X = np.max(np.abs(X_analitico - X_num))
    Error_Max_Y = np.max(np.abs(Y_analitico - Y_num))
    
    # Almacenar resultados
    results.append({
        'h': h, 
        'Error Máx. X(t)': Error_Max_X, 
        'Error Máx. Y(t)': Error_Max_Y
    })
    
    print(f"| h = {h:.<5g} | Puntos: {len(T_num):<6d} | Error X: {Error_Max_X:.8e} | Error Y: {Error_Max_Y:.8e} |")

print("-" * 70)
print("✅ Análisis de convergencia completado.")

# --- 4. Creación de la Tabla de Resultados ---

# Usando pandas para una tabla elegante y formateada
df_results = pd.DataFrame(results)

# Formateo de columnas para presentación
df_results['h'] = df_results['h'].apply(lambda x: f'{x:g}')
df_results['Error Máx. X(t)'] = df_results['Error Máx. X(t)'].apply(lambda x: f'{x:.6e}')
df_results['Error Máx. Y(t)'] = df_results['Error Máx. Y(t)'].apply(lambda x: f'{x:.6e}')

# 4. Crear una tabla bien formateada
print("\n📊 Tabla de Convergencia del Método RK4 para el Sistema (Orden O(h⁴))")
print("--------------------|-----------------------|-----------------------")
print(df_results.to_markdown(index=False, numalign="left", stralign="left"))
print("--------------------|-----------------------|-----------------------")
