import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Importamos pandas para una tabla formateada

# --- 1. Definición de Funciones (del script anterior) ---

def f(x, y):
    """EDO: dy/dx = -2*y + exp(-x)"""
    return -2.0 * y + np.exp(-x)

def y_analitica(x):
    """Solución Analítica: y(x) = exp(-x) - exp(-2*x)"""
    return np.exp(-x) - np.exp(-2.0 * x)

def RK4_solver(f, x0, y0, h, x_final):
    """Implementación del método Runge-Kutta de orden 4 (RK4)"""
    
    # Inicialización de las listas de resultados
    X = [x0]
    Y = [y0]

    x_n = x0
    y_n = y0

    # Bucle principal de integración
    while x_n < x_final:
        # Ajuste del último paso
        if x_n + h > x_final:
            h_step = x_final - x_n
            if h_step == 0:
                break
        else:
            h_step = h

        # Coeficientes de Runge-Kutta
        k1 = f(x_n, y_n)
        k2 = f(x_n + h_step/2.0, y_n + h_step/2.0 * k1)
        k3 = f(x_n + h_step/2.0, y_n + h_step/2.0 * k2)
        k4 = f(x_n + h_step, y_n + h_step * k3)

        # Cálculo de la nueva aproximación
        y_n_mas_1 = y_n + h_step/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # Actualización de valores
        x_n = x_n + h_step
        y_n = y_n_mas_1

        # Almacenamiento de resultados
        X.append(x_n)
        Y.append(y_n)

    return np.array(X), np.array(Y)

# --- 2. Parámetros y Bucle de Análisis ---

# Parámetros fijos de la EDO
x0 = 0.0          # Condición inicial x(0)
y0 = 0.0          # Condición inicial y(0)
x_final = 5.0     # Final del intervalo

# 1. Definir lista de tamaños de paso
h_values = [0.1, 0.01, 0.001]
results = []

print("🔬 Iniciando análisis de convergencia RK4...")
print("-" * 50)

# 2. Iterar sobre la lista de h
for h in h_values:
    # Ejecución del solver RK4
    X_rk4, Y_rk4 = RK4_solver(f, x0, y0, h, x_final)
    
    # Solución analítica en los mismos puntos X
    Y_analitica_h = y_analitica(X_rk4)
    
    # 3. Calcular el Error Máximo Absoluto
    Error_Absoluto = np.abs(Y_analitica_h - Y_rk4)
    Error_Maximo = np.max(Error_Absoluto)
    
    # Almacenar resultados
    results.append({'h': h, 'Error Global Máximo': Error_Maximo})
    
    print(f"| h = {h:.<5g} | Puntos calculados: {len(X_rk4):<6d} | Error Máximo: {Error_Maximo:.8e} |")

print("-" * 50)
print("✅ Análisis completado.")

# --- 3. Creación de la Tabla de Resultados ---

# Usando pandas para una tabla elegante
df_results = pd.DataFrame(results)
df_results['h'] = df_results['h'].apply(lambda x: f'{x:g}') # Formato para h
df_results['Error Global Máximo'] = df_results['Error Global Máximo'].apply(lambda x: f'{x:.6e}') # Formato para el error

# 4. Crear una tabla bien formateada
print("\n📊 Tabla de Convergencia del Método RK4")
print("-" * 50)
print(df_results.to_markdown(index=False, numalign="left", stralign="left"))
print("-" * 50)

# Opcional: Imprimir en formato de texto simple (para entornos sin pandas)
print("\nTabla de Convergencia (Texto Simple):")
print("{:<15} | {:<20}".format("Tamaño de Paso (h)", "Error Global Máximo"))
print("----------------|---------------------")
for r in results:
    print(f"{r['h']:<15g} | {r['Error Global Máximo']:.6e}")
    
