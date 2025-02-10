import cv2
import numpy as np

#Fazer o código do zero etapa por etapa

"""
✅ Detecta bordas através de gradientes direcionais.
✅ Analisa padrões de borda para definir a melhor suavização.
✅ Usa interpolação guiada pelo gradiente para minimizar aliasing.

Detecção de bordas usando gradiente direcional (Sobel).
Análise de padrões de borda para definir regiões de suavização.
Interpolação adaptativa para suavizar serrilhados sem borrar detalhes.

1️⃣ Detecção de Bordas:

Usa os filtros de Sobel para calcular gradiente horizontal e vertical.
Obtém a magnitude (intensidade) e orientação das bordas.
2️⃣ Criação de Máscara de Bordas:

Define um limiar (0.2) para criar um mapa de borda binário.
3️⃣ Suavização Direcionada:

Aplica um desfoque bilateral, que preserva bordas enquanto suaviza áreas internas.
Mescla os pixels da borda suavizada de volta na imagem original.


"""

def detect_edges(image):
    """ Detecta bordas usando gradientes direcionais (Sobel). """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gradientes na horizontal e vertical
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    cv2.imshow("grad X", grad_x)
    cv2.imshow("grad Y", grad_y)

    # Magnitude e orientação do gradiente
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    # Normalizar magnitude para [0,1]
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

    return magnitude, angle

def apply_smaa(image):
    """ Aplica um anti-aliasing baseado em bordas e interpolação. """
    edges, angles = detect_edges(image)

    # Criar um mapa binário das bordas
    edge_mask = (edges > 0.8).astype(np.uint8)
    cv2.imshow("Edge mask", edge_mask)

    # Aplicar um desfoque guiado para suavizar as bordas preservando detalhes
    blurred = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)

    # Interpolar pixels ao longo da orientação do gradiente
    result = image.copy()
    #for edge in edge_mask:
    #    if edge == 1:
    #        result[edge_mask] = blurred[edge_mask]

    return result

# Carregar imagem de teste
image = cv2.imread("images/game3.jpg")

# Aplicar SMAA
smaa_image = apply_smaa(image)

# Exibir resultado
cv2.imshow("Anti-Aliasing OFF", image)
cv2.imshow("SMAA Anti-Aliased", smaa_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
