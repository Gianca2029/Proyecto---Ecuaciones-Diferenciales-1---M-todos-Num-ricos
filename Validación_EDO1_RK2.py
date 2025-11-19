import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Implementación del Método de Heun (RK2) ---

def heun(f, y0, t):
    """
    Resuelve una EDO de primer orden y'(t) = f(t, y) usando el método de Heun (RK2).

    Parámetros:
    f : function
        La función que define la EDO: f(t, y).
    y0 : float
        La condición inicial, y(t[0]).
    t : np.ndarray
        Un array de puntos de tiempo donde se calculará la solución.

    Retorna:
    np.ndarray
        Un array con la solución numérica y(t).
    """
    N = len(t)
    y = np.zeros(N)
    y[0] = y0
    
    # El tamaño del paso h se determina a partir del array t
    # Asumimos que el paso es uniforme
    h = t[1] - t[0] 

    # Bucle principal del método
    for i in range(N - 1):
        # 1. Predictor (Método de Euler)
        k1 = f(t[i], y[i])
        
        # 2. Corrector (Heun)
        # Calcula el valor de y en t[i+1] usando la pendiente del predictor (k1)
        y_predictor = y[i] + h * k1
        
        # 3. Calcula la segunda pendiente (k2) en el punto predictor
        k2 = f(t[i+1], y_predictor)
        
        # 4. Cálculo final: Promedio de las pendientes (k1 y k2)
        y[i+1] = y[i] + (h / 2.0) * (k1 + k2)
        
    return y

# --- 2. Definición de la EDO y la Solución Analítica ---

def f_edo(t, y):
    """
    Función del lado derecho de la EDO: y'(t) = -2*y + exp(-t)
    """
    return -2.0 * y + np.exp(-t)

def solucion_analitica(x):
    """
    Solución analítica exacta de la EDO con y(0)=0: y(x) = exp(-x) - exp(-2x)
    """
    return np.exp(-x) - np.exp(-2.0 * x)

# --- 3. Ejecución y Comparación ---

# Parámetros
t0 = 0.0          # Tiempo inicial
y0 = 0.0          # Condición inicial y(0)
t_final = 5.0     # Tiempo final
h = 0.1           # Tamaño del paso

# Generación del array de puntos de tiempo
t_puntos = np.arange(t0, t_final + h, h) 

print(f"⚙️ Resolviendo EDO con el Método de Heun (RK2) en t=[0, 5], h={h}...")

# Ejecutar el método de Heun
y_num = heun(f_edo, y0, t_puntos)

# Calcular la solución analítica en los mismos puntos
y_ana = solucion_analitica(t_puntos)

print("✅ Solución numérica calculada.")
print("-" * 50)

# 4. Cálculo del Error Global Máximo
error_abs = np.abs(y_ana - y_num)
error_maximo = np.max(error_abs)

print(f"📈 Error Global Máximo (|y_analítica - y_numérica|) = {error_maximo:.8e}")
print("-" * 50)


## Imprimir los primeros 10 resultados en formato de tabla
datos_comparacion = pd.DataFrame({
    'Tiempo (t)': t_puntos,
    'Y Numérica (Heun)': y_num,
    'Y Analítica': y_ana,
    'Error Absoluto': error_abs
})

print("📋 Comparación de los primeros 10 puntos:")
print(datos_comparacion.head(10).to_markdown(index=False, floatfmt=".6f"))
print("-" * 50)


# 5. Generar Gráfico de Comparación
plt.figure(figsize=(10, 6))

# Solución analítica (línea continua)
plt.plot(t_puntos, y_ana, label='Solución Analítica $y(x)=e^{-x}-e^{-2x}$', 
         color='blue', linewidth=2)

# Solución numérica (puntos discretos)
plt.plot(t_puntos, y_num, 'ro', markersize=4, label=f'Solución Numérica (Heun, $h={h}$)', 
         alpha=0.6)

plt.title('Comparación: Método de Heun (RK2) vs Solución Analítica')
plt.xlabel('Tiempo $t$')
plt.ylabel('$y(t)$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
