import numpy as np
import pandas as pd

# --- 1. Funciones del Sistema ---

def f_edo(t, y):
    """
    Función del lado derecho de la EDO: y'(t) = -2*y + exp(-t)
    """
    return -2.0 * y + np.exp(-t)

def solucion_analitica(x):
    """
    Solución analítica exacta: y(x) = exp(-x) - exp(-2x)
    """
    return np.exp(-x) - np.exp(-2.0 * x)

def heun(f, y0, t):
    """
    Resuelve una EDO de primer orden usando el método de Heun (RK2).
    """
    N = len(t)
    y = np.zeros(N)
    y[0] = y0
    
    # Manejar el paso h, asegurando que se usa el paso real
    if N > 1:
        h = t[1] - t[0] 
    else:
        return y

    for i in range(N - 1):
        # Coeficiente k1 (Pendiente de Euler)
        k1 = f(t[i], y[i])
        
        # Valor del predictor (y_predictor)
        y_predictor = y[i] + h * k1
        
        # Coeficiente k2 (Pendiente en el punto predictor)
        k2 = f(t[i+1], y_predictor)
        
        # Corrector (Promedio de pendientes)
        y[i+1] = y[i] + (h / 2.0) * (k1 + k2)
        
    return y

# --- 2. Parámetros y Bucle de Iteración ---

# Parámetros fijos
x0 = 0.0          # Condición inicial x
y0 = 0.0          # Condición inicial y(0)
x_final = 2.0     # Final del intervalo

# Tamaños de paso a evaluar
h_values = [0.1, 0.01, 0.001]
results = []
errors = [] # Para almacenar errores para el cálculo del orden

print("🔬 Iniciando estudio de convergencia para el Método de Heun (RK2)...")
print("-" * 70)

# 1. Iterar sobre Tamaños de Paso
for h in h_values:
    # Generación del array de puntos de tiempo
    t_puntos = np.arange(x0, x_final + h, h)
    
    # 2. Ejecutar y calcular la solución
    y_num = heun(f_edo, y0, t_puntos)
    y_ana = solucion_analitica(t_puntos)
    
    # Cálculo del Error Global Máximo
    error_abs = np.abs(y_ana - y_num)
    error_maximo = np.max(error_abs)
    
    # Almacenar resultados
    results.append({
        'h': h, 
        'Puntos': len(t_puntos), 
        'Error Global Máximo': error_maximo
    })
    errors.append(error_maximo)
    
    print(f"| h = {h:.<5g} | Puntos: {len(t_puntos):<6d} | Error Máximo: {error_maximo:.8e} |")

print("-" * 70)
print("✅ Estudio de convergencia completado.")

# --- 3. Generación de la Tabla de Convergencia ---

# Usando pandas para formatear la tabla
df_results = pd.DataFrame(results)

# Formateo de columnas para presentación
df_results['h'] = df_results['h'].apply(lambda x: f'{x:g}')
df_results['Error Global Máximo'] = df_results['Error Global Máximo'].apply(lambda x: f'{x:.6e}')

print("\n📊 Tabla de Convergencia del Método de Heun (RK2)")
print("--------------------|----------|-----------------------")
print(df_results.to_markdown(index=False, numalign="left", stralign="left"))
print("--------------------|----------|-----------------------")


# --- 4. Estimación del Orden de Convergencia (p) ---

if len(errors) >= 2:
    # p = log(E1/E2) / log(h1/h2)
    E1, E2 = errors[0], errors[1] # Error para h=0.1 y h=0.01
    h1, h2 = h_values[0], h_values[1] # h=0.1 y h=0.01
    
    orden_observado = np.log(E1 / E2) / np.log(h1 / h2)
    
    print(f"\n🧠 Estimación del Orden de Convergencia (usando h={h1} y h={h2}):")
    print(f"Orden Observado (p): {orden_observado:.4f}")
