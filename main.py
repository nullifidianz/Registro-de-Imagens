from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import math

# =============================================
# CONSTANTES GLOBAIS E VARIÁVEIS DE ESTADO
# =============================================
N = 30           # Número de partículas
dt = 0.03        # Delta T (passo de tempo)
m = 0.3          # Massa de cada célula
K = 100          # Constante de elasticidade
Fat = 0.05       # Constante de atrito
g = np.array([0, 0, 10], dtype=float)  # Aceleração da gravidade
H = 100          # Altura inicial
T = 1            # Intervalo de tempo para timer
R = 5            # Raio da bolinha de colisão

# Variáveis de estado
contato = False  # Flag para contato com a esfera
quant_movimento = np.zeros(3)  # Energia cinética acumulada

# Arrays de estado do sistema
P = np.full((N, N, 3), H, dtype=float)  # Posições
A = np.zeros((N, N, 3))                 # Acelerações
V = np.zeros((N, N, 3))                 # Velocidades
Fatrito = np.zeros((N, N, 3))           # Forças de atrito
Felastica = np.zeros((N, N, 3))         # Forças elásticas

# =============================================
# FUNÇÕES DE INICIALIZAÇÃO
# =============================================
def setMesh():
    """Inicializa a malha de partículas"""
    global P
    for i in range(N):
        for j in range(N):
            P[i, j] = np.array([i, j, H], dtype=float)

def init():
    """Inicialização básica do OpenGL"""
    glEnable(GL_DEPTH_TEST)
    setMesh()

# =============================================
# CÁLCULOS FÍSICOS
# =============================================
def calcDistancia(i, j):
    """Calcula distância entre duas partículas"""
    return j - i

def calcHook():
    """Calcula forças elásticas usando a Lei de Hooke"""
    global Felastica
    for i in range(N):
        for j in range(N):
            f = np.zeros(3)
            # Vizinhos diretos (distância 1) e diagonais (distância sqrt(2))
            for di, dj, rest_length in [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                                      (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                                      (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    dist = calcDistancia(P[i, j], P[ni, nj])
                    f += dist * (np.linalg.norm(dist) - rest_length)
            Felastica[i, j] = f * K

def calcFat():
    """Calcula forças de atrito"""
    global Fatrito
    for i in range(N):
        for j in range(N):
            f = np.zeros(3)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if (di != 0 or dj != 0) and 0 <= ni < N and 0 <= nj < N:
                        f += V[ni, nj]
            Fatrito[i, j] = f * Fat

def calcPosicoes():
    """Calcula novas posições baseadas nas forças"""
    global contato, quant_movimento
    calcHook()
    calcFat()
    quant_movimento[:] = 0
    
    for i in range(N):
        for j in range(N):
            M1 = m
            # Calcula aceleração (F = ma)
            A[i, j] = (-g * M1 + Felastica[i, j] - Fatrito[i, j]) / (2 * M1)
            # Atualiza velocidade (v = v0 + a*dt)
            V[i, j] += A[i, j] * dt
            
            # Verifica colisão com a esfera
            if np.linalg.norm(P[i, j] - np.array([N//2, N//2, H*0.7])) > R + 1:
                # Atualiza posição (x = x0 + v*dt)
                P[i, j] += V[i, j] * dt
                quant_movimento += V[i, j]  # Acumula energia cinética
            else:
                contato = True

# =============================================
# FUNÇÕES DE RENDERIZAÇÃO
# =============================================
def DefineIluminacao():
    """Configura a iluminação da cena"""
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_DEPTH_TEST)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2]*3 + [1.0])
    
    # Configura duas fontes de luz
    for i, pos in enumerate([[0,250,0,1], [0,-250,0,1]]):
        glLightfv(GL_LIGHT0 + i, GL_POSITION, pos)
        glLightfv(GL_LIGHT0 + i, GL_AMBIENT, [0.2]*4)
        glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, [0.7]*4)
        glLightfv(GL_LIGHT0 + i, GL_SPECULAR, [0.5]*4)

def cor(forca):
    """Define a cor baseada na força aplicada"""
    forca /= 30.0
    difusa = [0.5*forca, 0.5*(1-forca), 1.0, 1.0]
    glMaterialfv(GL_FRONT, GL_DIFFUSE, difusa)
    difusa = [1.0, 0.5*forca, 0.5*(1-forca), 1.0]
    glMaterialfv(GL_BACK, GL_DIFFUSE, difusa)

def DesenhaMalha():
    """Renderiza a malha de tecido e a esfera de colisão"""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Configura a câmera
    gluLookAt(150, 250, 100, 0, H*0.7, 0, 0, 1, 0)
    
    DefineIluminacao()
    deslocamento = N / 2.0  # Centraliza a malha
    
    # Desenha o tecido como triângulos
    glBegin(GL_TRIANGLES)
    for i in range(N-1):
        for j in range(N-1):
            # Dois triângulos formando um quadrado
            for tri in [[(i,j),(i,j+1),(i+1,j)], [(i,j+1),(i+1,j+1),(i+1,j)]]:
                glColor3f(0.8, 0.8, 1.0)  # Cor base
                for vi,vj in tri:
                    cor(np.linalg.norm(Felastica[vi,vj]))  # Cor varia com a força
                    glVertex3f(P[vi,vj,0]-deslocamento, P[vi,vj,2], P[vi,vj,1]-deslocamento)
    glEnd()
    
    # Desenha a esfera de colisão
    glTranslatef(0, H*0.7, 0)
    glutSolidSphere(R, 10, 10)
    glutSwapBuffers()

# =============================================
# CONTROLE E LOOP PRINCIPAL
# =============================================
def massamola(v):
    """Função de timer para atualização da simulação"""
    calcPosicoes()
    glutPostRedisplay()  # Solicita redesenho
    
    # Continua a simulação se houver movimento ou sem contato
    if np.linalg.norm(quant_movimento) > 100 or not contato:
        glutTimerFunc(T, massamola, 1)

def AlteraTamanhoJanela(w, h):
    """Callback para redimensionamento de janela"""
    if h == 0: h = 1  # Evita divisão por zero
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(15.0, w/h, 0.5, 500.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def main():
    """Função principal que inicia a aplicação"""
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(600, 600)
    glutCreateWindow(b"Simulacao de Tecido - Python")
    
    init()
    glutDisplayFunc(DesenhaMalha)
    glutReshapeFunc(AlteraTamanhoJanela)
    glutTimerFunc(T, massamola, 1)
    glutMainLoop()

if __name__ == "__main__":
    main()