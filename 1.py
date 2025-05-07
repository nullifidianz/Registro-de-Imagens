import numpy as np

# Pontos da imagem original (x', y')
original_pts = np.array([
    [-1, 3],
    [5, 3],
    [-5, -3],
    [1, -3]
])

# Pontos da imagem deformada (x, y)
deformed_pts = np.array([
    [-3, 3],
    [3, 3],
    [-3, -3],
    [3, -3]
])

# Montar a matriz A e o vetor b
A = []
b = []

for i in range(4):
    x0, y0 = original_pts[i]
    x, y = deformed_pts[i]

    A.append([x0, y0, x0*y0, 1, 0, 0, 0, 0])  # equação para x
    A.append([0, 0, 0, 0, x0, y0, x0*y0, 1])  # equação para y
    b.append(x)
    b.append(y)

A = np.array(A)
b = np.array(b)

# Resolver sistema
C = np.linalg.solve(A, b)

# Separar constantes
C1, C2, C3, C4, C5, C6, C7, C8 = C

# Modelo de transformação
def deform(x0, y0):
    x = C1 * x0 + C2 * y0 + C3 * x0 * y0 + C4
    y = C5 * x0 + C6 * y0 + C7 * x0 * y0 + C8
    return (x, y)

# Pontos da imagem original a testar
test_pts = [
    (0.5, 3),
    (1.5, 3),
    (4.5, 3),
    (-0.5, 3),
    (-0.7, 3),
    (0.3, -3),
    (-4.3, -2),
    (-2.7, 0.5),
    (1.7, -2),
    (4.3, 2)
]

# Aplicar a deformação inversa
result_pts = [deform(x0, y0) for (x0, y0) in test_pts]

# Mostrar resultados
for i, ((x0, y0), (x, y)) in enumerate(zip(test_pts, result_pts)):
    print(f"Original ({x0:.2f}, {y0:.2f}) => Deformada ({x:.2f}, {y:.2f})")

