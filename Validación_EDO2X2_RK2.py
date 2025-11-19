import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definición del Sistema F(t, Y) ---

def F_sistema(t, Y):
    """
    Función del lado derecho del sistema de EDOs.
    
    Sistema: 
    x'(t) = 3*x + 4*y
    y'(t) = -4*x + 3*y
    
    Parámetros:
    t: Variable independiente (tiempo).
    Y: Vector de estado [x(t), y(t)].
    
    Retorna:
    Vector de derivadas [x'(t), y'(t)].
    """
    x = Y[0]
    y = Y[1]
    
    dxdt = 3.0 * x + 4.0 * y
    dydt = -4.0 * x + 3.0 * y
    
    # Retorna el vector de derivadas como un array NumPy
    return np.array([dxdt, dydt])

# --- 2. Solución Analítica ---

def x_analitico(t):
    """Solución analítica para x(t)."""
    # Solución original: x(t) = exp(3t) * cos(4t)
    # Nota: El usuario proporcionó la solución particular para y(0)=0:
    # x(t) = exp(3*t) * cos(4*t)
    # La solución general para y(0)=0 es simplemente x(t) = exp(3t) * cos(4t)
    return np.exp(3.0 * t) * np.cos(4.0 * t)

def y_analitico(t):
    """Solución analítica para y(t)."""
    # y(t) = -exp(3t) * sin(4t)
    return -np.exp(3.0 * t) * np.sin(4.0 * t)

# --- 3. Implementación del Método de Heun para Sistemas ---

def heun_sistema(F, Y0, t):
    """
    Resuelve un sistema de EDOs de 2x2 usando el Método de Heun (RK2) vectorial.
    
    Parámetros:
    F: La función vectorial que define el sistema F(t, Y).
    Y0: Vector de condiciones iniciales [x0, y0].
    t: Array de puntos de tiempo.
    
    Retorna:
    (X_num, Y_num) - Soluciones numéricas para x(t) e y(t).
    """
    N = len(t)
    
    # Inicialización del array de soluciones. Y_sol almacena [x, y] en cada paso.
    Y_sol = np.zeros((N, len(Y0)))
    Y_sol[0] = Y0
    
    # Tamaño del paso
    h = t[1] - t[0]

    # Bucle principal
    for i in range(N - 1):
        t_i = t[i]
        Y_i = Y_sol[i]
        
        # 1. Coeficiente k1 (Pendiente en t_i)
        # k1 = h * F(t_i, Y_i)
        k1 = h * F(t_i, Y_i)
        
        # 2. Coeficiente k2 (Pendiente en el punto predictor t_{i+1})
        # k2 = h * F(t_{i+1}, Y_i + k1)
        k2 = h * F(t[i+1], Y_i + k1)
        
        # 3. Cálculo final (Corrector: Promedio de k1 y k2)
        # Y_{i+1} = Y_i + 0.5 * (k1 + k2)
        Y_sol[i+1] = Y_i + 0.5 * (k1 + k2)
        
    # Desempaquetar las soluciones
    X_num = Y_sol[:, 0]
    Y_num = Y_sol[:, 1]
    
    return X_num, Y_num

# --- 4. Ejecución y Comparación ---

# Parámetros de la simulación
t0 = 0.0          # Tiempo inicial
Y0 = np.array([1.0, 0.0]) # Condiciones iniciales [x(0), y(0)]
t_final = 2.0     # Tiempo final
h = 0.01          # Tamaño del paso

# Generación del array de tiempo
t_puntos = np.arange(t0, t_final + h, h)

print(f"⚙️ Resolviendo Sistema de EDOs con Heun (RK2) en t=[0, 2], h={h}...")

# Ejecución del método de Heun vectorial
X_num, Y_num = heun_sistema(F_sistema, Y0, t_puntos)

# Cálculo de las soluciones analíticas
X_ana = x_analitico(t_puntos)
Y_ana = y_analitico(t_puntos)

print("✅ Soluciones numéricas calculadas.")
print("-" * 50)

# 5. Cálculo del Error Global Máximo
error_abs_X = np.abs(X_ana - X_num)
error_abs_Y = np.abs(Y_ana - Y_num)

error_maximo_X = np.max(error_abs_X)
error_maximo_Y = np.max(error_abs_Y)

print("📈 Errores Globales Máximos:")
print(f"  - Error Máximo para X(t): {error_maximo_X:.8e}")
print(f"  - Error Máximo para Y(t): {error_maximo_Y:.8e}")
print("-" * 50)


# 6. Visualización (Gráfico de Comparación)

plt.figure(figsize=(12, 6))

# Subplot para X(t)
plt.subplot(1, 2, 1)
plt.plot(t_puntos, X_ana, label='Analítica $x(t)$', color='blue', linewidth=2)
plt.plot(t_puntos, X_num, 'r--', markersize=2, label=f'Numérica Heun ($h={h}$)', alpha=0.7)
plt.title('Comparación $x(t)$ (Analítica vs Heun)')
plt.xlabel('Tiempo $t$')
plt.ylabel('$x(t)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Subplot para Y(t)
plt.subplot(1, 2, 2)
plt.plot(t_puntos, Y_ana, label='Analítica $y(t)$', color='green', linewidth=2)
plt.plot(t_puntos, Y_num, 'm--', markersize=2, label=f'Numérica Heun ($h={h}$)', alpha=0.7)
plt.title('Comparación $y(t)$ (Analítica vs Heun)')
plt.xlabel('Tiempo $t$')
plt.ylabel('$y(t)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# Gráfico del plano de fase (X vs Y) - Opcional, pero muy ilustrativo
plt.figure(figsize=(6, 6))
plt.plot(X_ana, Y_ana, label='Analítica (Plano de Fase)', color='black', linewidth=1)
plt.plot(X_num, Y_num, 'o', markersize=2, label='Numérica (Heun)', alpha=0.6)
plt.title('Plano de Fase $x(t)$ vs $y(t)$ (Espiral Divergente)')
plt.xlabel('$x(t)$')
plt.ylabel('$y(t)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.axis('equal') # Para que el círculo/espiral se vea correctamente
plt.show()
