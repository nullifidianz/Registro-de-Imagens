import numpy as np
import cv2


# Pontos da imagem distorcida (Figura 2-a)
image_points = np.array([
    [11, 16],  # x1, y1
    [1, 14],   # x2, y2
    [5, 4],    # x3, y3
    [19, 8]    # x4, y4
], dtype=np.float32)

# Pontos do mundo real (Figura 2-b) — marcador original
world_points = np.array([
    [8.8, 8.8],    # x1, y1
    [-8.8, 8.8],   # x2, y2
    [-8.8, -8.8],  # x3, y3
    [8.8, -8.8]    # x4, y4
], dtype=np.float32)

# Pontos da base da pirâmide (Figura 2-c)
pyramid_base = np.array([
    [-4.4, -4.4],
    [-4.4, 4.4],
    [4.4, 4.4],
    [4.4, -4.4]
], dtype=np.float32)

# Calcula a matriz de homografia H que mapeia world_points -> image_points
H, _ = cv2.findHomography(world_points, image_points)

# Aplica H aos pontos da base da pirâmide para obter a projeção na imagem
'''
Aplica uma transformação de perspectiva (homografia) aos pontos da base da pirâmide definidos no mundo real.

Etapas:
1. pyramid_base.reshape(-1, 1, 2):
   - Reorganiza o array de shape (N, 2) para (N, 1, 2), formato exigido pela função cv2.perspectiveTransform().
   - Esse formato representa N pontos 2D, cada um com uma dimensão extra, por exigência da API do OpenCV.

2. cv2.perspectiveTransform(..., H):
   - Aplica a matriz de homografia H, obtida anteriormente, que mapeia pontos do plano real para a imagem.
   - Para cada ponto (x0, y0), aplica a transformação projetiva:
        [x']   = H * [x0]
        [y']           [y0]
        [w ]           [1 ]
     Depois normaliza para coordenadas de imagem: (x, y) = (x'/w, y'/w)

3. .reshape(-1, 2):
   - Remove a dimensão extra, convertendo o resultado de shape (N, 1, 2) para (N, 2).
   - Isso gera uma lista simples de pontos 2D transformados, prontos para uso (exibição, plotagem, etc).

Resultado final: pyramid_base_homog contém as coordenadas dos pontos da base da pirâmide, como seriam vistos na imagem com a mesma distorção de perspectiva do marcador.
'''

pyramid_base_homog = cv2.perspectiveTransform(pyramid_base.reshape(-1, 1, 2), H).reshape(-1, 2)

# Mostra os pontos projetados
print("Pontos projetados da base da pirâmide na imagem:")
for i, (x, y) in enumerate(pyramid_base_homog):
    print(f"P{i+1}: ({x:.2f}, {y:.2f})")
