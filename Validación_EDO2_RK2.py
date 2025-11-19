import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definición del Sistema F(x, Y) ---

def F_sistema(x, Y):
    """
    Función del lado derecho del sistema de EDOs.
    y1' = y2, y2' = sin(x) - y1
    """
    y1 = Y[0]
    
    dy1dx = Y[1]            
    dy2dx = np.sin(x) - y1  
    
    return np.array([dy1dx, dy2dx])

# --- 2. Solución Analítica ---

def solucion_analitica(x):
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

# --- 4. Ejecución y Gráfica ---

# Parámetros de la simulación
x0 = 0.0          
Y0 = np.array([0.0, 0.0]) 
x_final = 4 * np.pi 
h = 0.1           # Paso fijo

# Generación del array de tiempo
x_puntos = np.arange(x0, x_final + h, h)

print(f"⚙️ Resolviendo EDO de 2do Orden con Heun (RK2).")
print(f"Intervalo: [0, {x_final:.4f}], Paso h={h}")

# Ejecución del método de Heun vectorial
Y1_num = heun_sistema(F_sistema, Y0, x_puntos)

# Cálculo de la solución analítica
Y1_ana = solucion_analitica(x_puntos)

# Cálculo y reporte del Error Global Máximo
error_maximo = np.max(np.abs(Y1_ana - Y1_num))
print("-" * 60)
print(f"✅ Solución calculada. Error Global Máximo: {error_maximo:.8e}")
print("-" * 60)

# Generación del Gráfico
plt.figure(figsize=(10, 6))

# Solución analítica (línea continua y gruesa)
plt.plot(x_puntos, Y1_ana, label='Solución Analítica $y(x)$ (Resonancia)', 
         color='blue', linewidth=3, alpha=0.6)

# Solución numérica (puntos discretos)
plt.plot(x_puntos, Y1_num, 'ro', markersize=3, label=f'Solución Numérica (Heun, $h={h}$)', 
         alpha=0.9)

plt.title('Comparación: Método de Heun (RK2) para EDO de 2do Orden con Resonancia')
plt.xlabel('Variable $x$')
plt.ylabel('$y(x)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()
