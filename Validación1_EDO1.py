import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definición de Funciones ---

def f(x, y):
    """
    Función del lado derecho de la EDO: dy/dx = -2*y + exp(-x)
    """
    # Usamos np.exp para que funcione con arrays si es necesario
    return -2.0 * y + np.exp(-x)

def y_analitica(x):
    """
    Solución analítica (correcta) de la EDO: y(x) = exp(-x) - exp(-2*x)
    """
    return np.exp(-x) - np.exp(-2.0 * x)

# --- 2. Implementación del Solver RK4 ---

def RK4_solver(f, x0, y0, h, x_final):
    """
    Implementa el método Runge-Kutta de orden 4 (RK4) para una EDO de primer orden.

    Parámetros:
    f: Función f(x, y) que define la EDO (dy/dx = f(x, y)).
    x0: Condición inicial para x.
    y0: Condición inicial para y.
    h: Tamaño del paso.
    x_final: Valor final de x para la integración.

    Retorna:
    (X, Y) - Arrays de las coordenadas x e y de la solución numérica.
    """
    # Inicialización de las listas de resultados
    X = [x0]
    Y = [y0]

    # Inicialización del punto actual
    x_n = x0
    y_n = y0

    # Bucle principal de integración
    while x_n < x_final:
        # Aseguramos que el último paso no exceda x_final
        if x_n + h > x_final:
            h = x_final - x_n
            if h == 0:
                break

        # Coeficientes de Runge-Kutta
        # 
        k1 = f(x_n, y_n)
        k2 = f(x_n + h/2.0, y_n + h/2.0 * k1)
        k3 = f(x_n + h/2.0, y_n + h/2.0 * k2)
        k4 = f(x_n + h, y_n + h * k3)

        # Cálculo de la nueva aproximación
        y_n_mas_1 = y_n + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # Actualización de valores
        x_n = x_n + h
        y_n = y_n_mas_1

        # Almacenamiento de resultados
        X.append(x_n)
        Y.append(y_n)

    return np.array(X), np.array(Y)

# --- 3. Parámetros y Ejecución ---

# Parámetros de la EDO
x0 = 0.0          # Condición inicial x(0)
y0 = 0.0          # Condición inicial y(0)
x_final = 5.0     # Final del intervalo
h = 0.1           # Tamaño del paso

print(f"⚙️ Iniciando Solución RK4 con h = {h} en el intervalo [0, {x_final}]...")

# Ejecución del solver RK4
X_rk4, Y_rk4 = RK4_solver(f, x0, y0, h, x_final)

# Cálculo de la solución analítica para los mismos puntos x
Y_analitica = y_analitica(X_rk4)

print("✅ Solución numérica calculada.")
print("-" * 40)

# --- 4. Cálculo del Error y Validación ---

# Error absoluto en cada punto
Error_Absoluto = np.abs(Y_analitica - Y_rk4)

# Error global máximo
Error_Maximo = np.max(Error_Absoluto)

print(f"📈 Error Global Máximo (RK4 vs Analítica) = {Error_Maximo:.8e}")
print("-" * 40)

# --- 5. Visualización de Resultados (Gráfica) ---

plt.figure(figsize=(10, 6))

# Gráfica de la solución analítica
plt.plot(X_rk4, Y_analitica, label='Solución Analítica: $e^{-x} - e^{-2x}$', color='blue', linewidth=2)

# Gráfica de la solución numérica (puntos para h=0.1)
plt.plot(X_rk4, Y_rk4, 'ro', markersize=3, label=f'Solución Numérica (RK4, $h={h}$)', alpha=0.6)

plt.title('Comparación: Solución RK4 vs Solución Analítica')
plt.xlabel('$x$')
plt.ylabel('$y(x)$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- 6. Gráfica Adicional de Error ---

plt.figure(figsize=(10, 3))
plt.plot(X_rk4, Error_Absoluto, label='Error Absoluto $|y_{analitica} - y_{RK4}|$', color='red')
plt.plot(X_rk4, np.full_like(X_rk4, Error_Maximo), 'k--', label=f'Error Máximo: {Error_Maximo:.2e}')
plt.title('Error Absoluto en la Solución Numérica')
plt.xlabel('$x$')
plt.ylabel('Error Absoluto')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
