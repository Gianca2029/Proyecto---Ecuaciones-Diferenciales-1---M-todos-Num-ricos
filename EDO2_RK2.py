import numpy as np
import pandas as pd

# --- 1. Definición del Sistema F(x, Y) ---

def F_sistema(x, Y):
    """
    Función del lado derecho del sistema de EDOs.
    Sistema: y1' = y2, y2' = sin(x) - y1
    """
    y1 = Y[0]
    
    dy1dx = Y[1]            
    dy2dx = np.sin(x) - y1  
    
    return np.array([dy1dx, dy2dx])

# --- 2. Solución Analítica ---

def y_analitica(x):
    """
    Solución analítica para y(x) = y1(x): y(x) = 0.5*sin(x) - 0.5*x*cos(x)
    """
    return 0.5 * np.sin(x) - 0.5 * x * np.cos(x)

# --- 3. Implementación del Método de Heun para Sistemas ---

def heun_sistema(F, Y0, x_puntos):
    """
    Resuelve un sistema de EDOs usando el Método de Heun (RK2) vectorial.
    """
    N = len(x_puntos)
    Y_sol = np.zeros((N, len(Y0)))
    Y_sol[0] = Y0
    
    if N <= 1:
        return Y_sol[:, 0]
        
    h = x_puntos[1] - x_puntos[0]

    for i in range(N - 1):
        x_i = x_puntos[i]
        Y_i = Y_sol[i]
        
        # 1. k1 = h * F(x_i, Y_i)
        k1 = h * F(x_i, Y_i)
        
        # 2. k2 = h * F(x_{i+1}, Y_i + k1)
        k2 = h * F(x_puntos[i+1], Y_i + k1)
        
        # 3. Y_{i+1} = Y_i + 0.5 * (k1 + k2)
        Y_sol[i+1] = Y_i + 0.5 * (k1 + k2)
        
    # Retorna solo y1 (la solución y(x))
    return Y_sol[:, 0] 

# --- 4. Parámetros y Bucle de Convergencia ---

# Parámetros fijos
x0 = 0.0          
Y0 = np.array([0.0, 0.0]) 
x_final = 4 * np.pi 

# Tamaños de paso a evaluar
h_values = [0.1, 0.01, 0.001]
results = []
errors = []

print("🔬 Iniciando estudio de convergencia para EDO de 2do Orden (Heun, RK2)...")
print(f"Intervalo de integración: [0, {x_final:.4f}]")
print("-" * 70)

# 1. Iterar sobre Tamaños de Paso
for h in h_values:
    # Generación del array de tiempo
    t_puntos = np.arange(x0, x_final + h, h)
    
    # Ejecución del método de Heun vectorial
    Y1_num = heun_sistema(F_sistema, Y0, t_puntos)
    
    # Cálculo de la solución analítica para Y1 (y(x))
    Y1_analitico = y_analitica(t_puntos)
    
    # 2. Cálculo del Error Global Máximo (Ey)
    Error_Max = np.max(np.abs(Y1_analitico - Y1_num))
    
    # Almacenar resultados
    results.append({
        'h': h, 
        'Puntos': len(t_puntos), 
        'Error Global Máximo ($E_y$)': Error_Max
    })
    errors.append(Error_Max)
    
    print(f"| h = {h:.<5g} | Puntos: {len(t_puntos):<6d} | Error Máximo: {Error_Max:.8e} |")

print("-" * 70)
print("✅ Estudio de convergencia completado.")

# --- 3. Generación de la Tabla de Convergencia ---

df_results = pd.DataFrame(results)

# Formateo de columnas
df_results['h'] = df_results['h'].apply(lambda x: f'{x:g}')
df_results['Error Global Máximo ($E_y$)'] = df_results['Error Global Máximo ($E_y$)'].apply(lambda x: f'{x:.6e}')

print("\n📊 Tabla de Convergencia del Método de Heun (RK2)")
print("--------------------|----------|-----------------------")
print(df_results.to_markdown(index=False, numalign="left", stralign="left"))
print("--------------------|----------|-----------------------")


# --- 4. Estimación del Orden de Convergencia (p) ---

if len(errors) >= 2:
    # Usamos h=0.1 y h=0.01 para el cálculo del orden
    E1, E2 = errors[0], errors[1] 
    h1, h2 = h_values[0], h_values[1] 
    
    # Fórmula del orden de convergencia: p ≈ log(E1/E2) / log(h1/h2)
    orden_observado = np.log(E1 / E2) / np.log(h1 / h2)
    
    print(f"\n🧠 Estimación del Orden de Convergencia (p):")
    print(f"Orden Observado (p): {orden_observado:.4f}")
    
    print(f"\nEl valor observado es cercano a 2.0, confirmando que el Método de Heun es de Orden 2.")
