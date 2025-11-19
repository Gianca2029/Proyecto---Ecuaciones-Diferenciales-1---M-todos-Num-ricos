import numpy as np
import pandas as pd

# --- 1. Definición del Sistema F(t, Y) ---

def F_sistema(t, Y):
    """
    Función del lado derecho del sistema de EDOs.
    Sistema: x'(t) = 3*x + 4*y, y'(t) = -4*x + 3*y
    """
    x = Y[0]
    y = Y[1]
    
    dxdt = 3.0 * x + 4.0 * y
    dydt = -4.0 * x + 3.0 * y
    
    return np.array([dxdt, dydt])

# --- 2. Solución Analítica ---

def x_analitico(t):
    """Solución analítica para x(t)."""
    # x(t) = exp(3t) * cos(4t)
    return np.exp(3.0 * t) * np.cos(4.0 * t)

def y_analitico(t):
    """Solución analítica para y(t)."""
    # y(t) = -exp(3t) * sin(4t)
    return -np.exp(3.0 * t) * np.sin(4.0 * t)

# --- 3. Implementación del Método de Heun para Sistemas ---

def heun_sistema(F, Y0, t):
    """
    Resuelve un sistema de EDOs de 2x2 usando el Método de Heun (RK2) vectorial.
    """
    N = len(t)
    Y_sol = np.zeros((N, len(Y0)))
    Y_sol[0] = Y0
    
    if N <= 1:
        return Y_sol[:, 0], Y_sol[:, 1]
        
    h = t[1] - t[0]

    for i in range(N - 1):
        t_i = t[i]
        Y_i = Y_sol[i]
        
        # k1 = h * F(t_i, Y_i)
        k1 = h * F(t_i, Y_i)
        
        # k2 = h * F(t_{i+1}, Y_i + k1)
        k2 = h * F(t[i+1], Y_i + k1)
        
        # Y_{i+1} = Y_i + 0.5 * (k1 + k2)
        Y_sol[i+1] = Y_i + 0.5 * (k1 + k2)
        
    X_num = Y_sol[:, 0]
    Y_num = Y_sol[:, 1]
    
    return X_num, Y_num

# --- 4. Parámetros y Bucle de Convergencia ---

# Parámetros fijos
t0 = 0.0          # Tiempo inicial
Y0 = np.array([1.0, 0.0]) # Condiciones iniciales [x(0), y(0)]
t_final = 2.0     # Final del intervalo

# Tamaños de paso a evaluar
h_values = [0.1, 0.01, 0.001]
results = []
errors_X = []
errors_Y = []

print("🔬 Iniciando estudio de convergencia para el Método de Heun (RK2) en Sistemas...")
print("-" * 75)

# 1. Iterar sobre Tamaños de Paso
for h in h_values:
    # Generación del array de tiempo
    t_puntos = np.arange(t0, t_final + h, h)
    
    # Ejecución del método de Heun vectorial
    X_num, Y_num = heun_sistema(F_sistema, Y0, t_puntos)
    
    # Cálculo de las soluciones analíticas
    X_ana = x_analitico(t_puntos)
    Y_ana = y_analitico(t_puntos)
    
    # 2. Cálculo del Error Global Máximo
    Error_Max_X = np.max(np.abs(X_ana - X_num))
    Error_Max_Y = np.max(np.abs(Y_ana - Y_num))
    
    # Almacenar resultados
    results.append({
        'h': h, 
        'Puntos': len(t_puntos), 
        'Error Máximo (X)': Error_Max_X,
        'Error Máximo (Y)': Error_Max_Y
    })
    errors_X.append(Error_Max_X)
    errors_Y.append(Error_Max_Y)
    
    print(f"| h = {h:.<5g} | Puntos: {len(t_puntos):<6d} | Error X: {Error_Max_X:.8e} | Error Y: {Error_Max_Y:.8e} |")

print("-" * 75)
print("✅ Estudio de convergencia completado.")

# --- 3. Generación de la Tabla de Convergencia ---

df_results = pd.DataFrame(results)

# Formateo de columnas para presentación
df_results['h'] = df_results['h'].apply(lambda x: f'{x:g}')
df_results['Error Máximo (X)'] = df_results['Error Máximo (X)'].apply(lambda x: f'{x:.6e}')
df_results['Error Máximo (Y)'] = df_results['Error Máximo (Y)'].apply(lambda x: f'{x:.6e}')

print("\n📊 Tabla de Convergencia del Método de Heun (RK2) para el Sistema 2x2")
print("--------------------|----------|-----------------------|-----------------------")
print(df_results.to_markdown(index=False, numalign="left", stralign="left"))
print("--------------------|----------|-----------------------|-----------------------")


# --- 4. Estimación del Orden de Convergencia (p) ---

if len(errors_X) >= 2 and len(errors_Y) >= 2:
    # Usamos h=0.1 y h=0.01
    h1, h2 = h_values[0], h_values[1]
    
    # Cálculo para X(t)
    E1_X, E2_X = errors_X[0], errors_X[1] 
    orden_X = np.log(E1_X / E2_X) / np.log(h1 / h2)
    
    # Cálculo para Y(t)
    E1_Y, E2_Y = errors_Y[0], errors_Y[1] 
    orden_Y = np.log(E1_Y / E2_Y) / np.log(h1 / h2)
    
    print(f"\n🧠 Estimación del Orden de Convergencia (p) (usando h={h1} y h={h2}):")
    print(f"  - Orden Observado para X(t): {orden_X:.4f}")
    print(f"  - Orden Observado para Y(t): {orden_Y:.4f}")
    
    # Promedio
    orden_promedio = (orden_X + orden_Y) / 2
    print(f"\nOrden Teórico Confirmado: ~2.0 (Promedio Observado: {orden_promedio:.4f})")
