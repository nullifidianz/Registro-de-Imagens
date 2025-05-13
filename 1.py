import numpy as np

# Pontos na imagem original
original_points = np.array([
    [-1,  3],
    [ 5,  3],
    [-5, -3],
    [ 1, -3]
])

# Pontos na imagem deformada
deformed_points = np.array([
    [-3,  3],
    [ 3,  3],
    [-3, -3],
    [ 3, -3]
])

# Montar matriz A
A = []
for x0, y0 in original_points:
    A.append([x0, y0, x0 * y0, 1])
A = np.array(A)

# Vetores bx e by (deformada)
bx = deformed_points[:, 0]
by = deformed_points[:, 1]

# Resolver o sistema A * Cx = bx e A * Cy = by
Cx = np.linalg.solve(A, bx)
Cy = np.linalg.solve(A, by)

# Mostrar coeficientes
print("Coeficientes para x:")
for i, c in enumerate(Cx, 1):
    print(f"C{i} = {c:.6f}")
print("\nCoeficientes para y:")
for i, c in enumerate(Cy, 5):
    print(f"C{i} = {c:.6f}")

# Agora aplicar nos novos pontos
new_original_points = np.array([
    [ 0.5,  3],
    [ 1.5,  3],
    [ 4.5,  3],
    [-0.5,  3],
    [-0.7,  3],
    [ 0.3, -3],
    [-4.3, -2],
    [-2.7,  0.5],
    [ 1.7, -2],
    [ 4.3,  2]
])

# Aplicar o modelo
def apply_model(points, Cx, Cy):
    results = []
    for x0, y0 in points:
        x = Cx[0]*x0 + Cx[1]*y0 + Cx[2]*x0*y0 + Cx[3]
        y = Cy[0]*x0 + Cy[1]*y0 + Cy[2]*x0*y0 + Cy[3]
        results.append((x, y))
    return results

recovered_points = apply_model(new_original_points, Cx, Cy)

print("\nPontos recuperados na imagem deformada:")
for i, (x, y) in enumerate(recovered_points, 1):
    print(f"{i:2d}: ({x:.3f}, {y:.3f})")
