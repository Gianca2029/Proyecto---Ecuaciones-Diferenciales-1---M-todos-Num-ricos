import numpy as np
import pandas as pd

# --- 1. Definición de Funciones ---

def F_sistema_segundo_orden(x, Y):
    """
    Función del lado derecho del sistema de EDOs:
    y1' = y2
    y2' = sin(x) - y1
    """
    y1 = Y[0]
    dy1dx = Y[1]            # y1' = y2
    dy2dx = np.sin(x) - y1  # y2' = sin(x) - y1
    return np.array([dy1dx, dy2dx])

def y_analitica(x):
    """
    Solución analítica para y(x) = 0.5 * sin(x) - 0.5 * x * cos(x)
    """
    return 0.5 * np.sin(x) - 0.5 * x * np.cos(x)

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
    
    # Retorna T_arr y las dos columnas (Y1 y Y2)
    return T_arr, Y_sol_arr[:, 0], Y_sol_arr[:, 1]

# --- 2. Parámetros y Bucle de Análisis ---

# Parámetros fijos
x0 = 0.0          
Y0 = [0.0, 0.0]   
x_final = 20.0    # Intervalo largo para ver el efecto de resonancia

# 1. Definir lista de tamaños de paso
h_values = [0.1, 0.01, 0.001]
results = []

print("🔬 Iniciando estudio de convergencia para EDO de 2do Orden (RK4)...")
print("-" * 70)

# 2. Iterar sobre la lista de h
for h in h_values:
    # Ejecución del solver RK4
    X_num, Y1_num, Y2_num = RK4_solver_sistema(F_sistema_segundo_orden, x0, Y0, h, x_final)
    
    # Solución analítica para Y1 (y(x)) en los mismos puntos X
    Y1_analitico = y_analitica(X_num)
    
    # 3. Calcular el Error Máximo Absoluto para Y1(x)
    Error_Abs_Y1 = np.abs(Y1_analitico - Y1_num)
    Error_Max_Y1 = np.max(Error_Abs_Y1)
    
    # Almacenar resultados
    results.append({'h': h, 'Error Global Máximo': Error_Max_Y1})
    
    print(f"| h = {h:.<5g} | Puntos: {len(X_num):<6d} | Error Máximo (Y1): {Error_Max_Y1:.8e} |")

print("-" * 70)
print("✅ Análisis de convergencia completado.")

# --- 3. Creación de la Tabla de Resultados ---

# Usando pandas para una tabla elegante y formateada
df_results = pd.DataFrame(results)

# Formateo de columnas para presentación
df_results['h'] = df_results['h'].apply(lambda x: f'{x:g}')
df_results['Error Global Máximo'] = df_results['Error Global Máximo'].apply(lambda x: f'{x:.6e}')

# 4. Generar tabla formateada
print("\n📊 Tabla de Convergencia del Método RK4 para la EDO de 2do Orden")
print("--------------------|-----------------------")
print(df_results.to_markdown(index=False, numalign="left", stralign="left"))
print("--------------------|-----------------------")
