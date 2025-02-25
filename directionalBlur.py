import cv2
import numpy as np

#CONCLUSÃO: 
"""
Acho que o gaussian 3x3 suaviza melhor em áreas planas, porém em regiões circulares ou diagonais, a janela 3x3 tende a ficar marcante e perceptível
O que não ocorre no blur direcionado, dado que as diagonais também são uma referência para a suavização

COMPARAÇÃO COM FXAA
Ela não incorpora todas as otimizações e detalhes finos do algoritmo FXAA original,
como a amostragem adaptativa com pesos calculados de forma contínua ou o ajuste dinâmico dos parâmetros de blending
"""

#Kernels para fazer a média entre os pixels vizinhos em cada direção
vertical_kernel = np.array([[0, 1/3, 0], [0, 1/3, 0], [0, 1/3, 0]], dtype=np.float32)
horizontal_kernel = np.array([[0, 0, 0], [1/3, 1/3, 1/3], [0, 0, 0]], dtype=np.float32)
diag_DE_kernel = np.array([[0, 0, 1/3], [0, 1/3, 0], [1/3, 0, 0]], dtype=np.float32)
diag_ED_kernel = np.array([[1/3, 0, 0], [0, 1/3, 0], [0, 0, 1/3]], dtype=np.float32)

vertical_kernel_gauss = np.array([[0, 0.25, 0], [0, 0.5, 0], [0, 0.25, 0]], dtype=np.float32)
horizontal_kernel_gauss = np.array([[0, 0, 0], [0.25, 0.5, 0.25], [0, 0, 0]], dtype=np.float32)
diag_DE_kernel_gauss = np.array([[0, 0, 0.25], [0, 0.5, 0], [0.25, 0, 0]], dtype=np.float32)
diag_ED_kernel_gauss = np.array([[0.25, 0, 0], [0, 0.5, 0], [0, 0, 0.25]], dtype=np.float32)

def directional_blur(L_img, y, x, kernel, kernel_gaussian):
    sum = 0
    sum_gaussian = 0
    h, w = L_img.shape
    for row in range(-1, 2):
        for col in range(-1, 2):
            y_idx, x_idx = y + row, x + col
            if 0 <= y_idx < h and 0 <= x_idx < w: #Verify if its a valid image index 
                sum += L_img[y_idx, x_idx] * kernel[row+1, col+1]
                sum_gaussian += L_img[y_idx, x_idx] * kernel_gaussian[row+1, col+1]  
    final_L_directed_blur[y, x] = np.clip(sum, 0, 255).astype(np.uint8)
    final_L_directed_blur_gaussian[y, x] = np.clip(sum_gaussian, 0, 255).astype(np.uint8)

def find_high_mag_edges(L_img, angles, magnitudes, threshold=51):
    L_img_blur = cv2.GaussianBlur(L_img, (3, 3), 0)

    index_high_mag = np.where(magnitudes > threshold)
    for y, x in zip(index_high_mag[0], index_high_mag[1]):
        #Gaussian filter AA
        final_L_gaussian[y,x] = L_img_blur[y,x]
        
        #Aplica directed blur AA 
        #Borda Vertical -> Ângulos da matiz (0 a 180) que representam orientação horizontal 
        if ((angles[y,x] >= 0 and angles[y,x] < 12) or (angles[y,x] >= 79 and angles[y,x] < 102) or (angles[y,x] >= 169 and angles[y,x] <= 180)):
            directional_blur(L_img, y, x, horizontal_kernel, horizontal_kernel_gauss)
        #Borda Horizontal -> Ângulos da matiz (0 a 180) que representam orientação vertical
        elif ((angles[y,x] >= 34 and angles[y,x] < 57) or (angles[y,x] >= 124 and angles[y,x] < 147)):
            directional_blur(L_img, y, x, vertical_kernel, vertical_kernel_gauss)
        #Borda Diag -> ESQUERDA PRA DIREITA -> Ângulos da matiz (0 a 180) que representam orientação DIAG DE
        elif ((angles[y,x] >= 12 and angles[y,x] < 34) or (angles[y,x] >= 102 and angles[y,x] < 124)):
            directional_blur(L_img, y, x, diag_DE_kernel, diag_DE_kernel_gauss)
        #Borda Diag -> DIREITA PRA ESQUERDA -> Ângulos da matiz (0 a 180) que representam orientação DIAG ED
        elif ((angles[y,x] >= 57 and angles[y,x] < 79) or (angles[y,x] >= 147 and angles[y,x] < 169)):
            directional_blur(L_img, y, x, diag_ED_kernel, diag_ED_kernel_gauss)

#img = cv2.imread("images/game3.jpg")
img = cv2.imread("images/ts_dino.png")
#img = cv2.imread("images/GT2.bmp")
#img = cv2.imread("images/head_TheWitcher.jpg")

#img = cv2.imread("images/mario_zoom.png")
#img = cv2.imread("images/mario.png")
#img = cv2.imread("images/TheWitcher_serr.jpg")


if img is None:
    raise ValueError("Erro ao abrir a imagem.")

height, width, c = img.shape
NN_RESIZE = 2

DIRECTORY = "ts_mesa"
#DIRECTORY = "Carros"
#DIRECTORY = "Witcher"
#DIRECTORY = "game_3"
#DIRECTORY = "mario_zoom"
#DIRECTORY = "mario"
#DIRECTORY = "witcher_full"

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Extrai gradientes com filtro de Sobel
grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

#Calcula magnitude e orientação do gradiente
magnitude = np.sqrt(grad_x**2 + grad_y**2)
angle = np.arctan2(grad_y, grad_x)
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
red_magnitude = cv2.resize(magnitude, (width*NN_RESIZE, height*NN_RESIZE), interpolation=cv2.INTER_NEAREST)
angle_hue = ((angle + np.pi) * (180 / (2 * np.pi))).astype(np.uint8)

#Manipulações para exibir a orientação em HSL
height, width, ch = img.shape
orientacao = np.zeros((height, width, 3), dtype=np.uint8)
orientacao[:, :, 0] = angle_hue #Hue
orientacao[:, :, 2] = 255 #S
orientacao[:, :, 1] = (0.5 * magnitude).astype(np.uint8) #L
bgr_image = cv2.cvtColor(orientacao, cv2.COLOR_HLS2BGR)
red_bgr_img = cv2.resize(bgr_image, (width*NN_RESIZE, height*NN_RESIZE), interpolation=cv2.INTER_NEAREST)

#Exibe orientação e magnitude
cv2.imshow("Orientacao", red_bgr_img)
cv2.imwrite(f"{DIRECTORY}/Orientacao.jpg", red_bgr_img)
cv2.imshow("magnitude", red_magnitude/255)
cv2.imwrite(f"{DIRECTORY}/Magnitude.jpg", red_magnitude)


#APLICAR ANTI-ALIASING
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
H, L, S = cv2.split(img_hls)

#Copia canal da Luminância para manipulações
final_L_gaussian = L.copy()
final_L_directed_blur = L.copy()
final_L_directed_blur_gaussian = L.copy()

#Encontra bordas (altas magnitudes)
find_high_mag_edges(L, angle_hue, magnitude)

#Exibe imagem final com directed blur anti-aliasing
final_blur_HLS = cv2.merge([H, final_L_directed_blur, S])
final_blur = cv2.cvtColor(final_blur_HLS, cv2.COLOR_HLS2BGR)
red_final_blur = cv2.resize(final_blur, (width*NN_RESIZE, height*NN_RESIZE), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Directed blur", red_final_blur/255)
cv2.imwrite(f"{DIRECTORY}/3 DirectedBlur.jpg", red_final_blur)

#Exibe imagem final com directed blur anti-aliasing -> com kernel gaussiano
final_blur_HLS_gaussian = cv2.merge([H, final_L_directed_blur_gaussian, S])
final_blur_gaussian = cv2.cvtColor(final_blur_HLS_gaussian, cv2.COLOR_HLS2BGR)
red_final_blur_gaussian = cv2.resize(final_blur_gaussian, (width*NN_RESIZE, height*NN_RESIZE), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Directed blur gaussian kernel", red_final_blur_gaussian/255)
cv2.imwrite(f"{DIRECTORY}/3 DirectedBlur-gaussianKernel.jpg", red_final_blur_gaussian)

#Exibe imagem final com gaussian filter anti-aliasing
final_gaussian_HLS = cv2.merge([H, final_L_gaussian, S])
final_gaussian = cv2.cvtColor(final_gaussian_HLS, cv2.COLOR_HLS2BGR)
red_final_gaussian = cv2.resize(final_gaussian, (width*NN_RESIZE, height*NN_RESIZE), interpolation=cv2.INTER_NEAREST)
cv2.imshow("gaussian", red_final_gaussian/255)
cv2.imwrite(f"{DIRECTORY}/1 gaussian.jpg", red_final_gaussian)

#Exibe bordas selecionadas para suavização
blurred_edges = final_L_directed_blur - L
blurred_edges_norm = cv2.normalize(blurred_edges, None, 0, 255, cv2.NORM_MINMAX)
red_blurred_edges_norm = cv2.resize(blurred_edges_norm, (width*NN_RESIZE, height*NN_RESIZE), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Bordas identificadas para suavizar", red_blurred_edges_norm/255)
cv2.imwrite(f"{DIRECTORY}/Bordas identificadas para suavizar.jpg", red_blurred_edges_norm)

#Exibe imagem original
red_img = cv2.resize(img, (width*NN_RESIZE, height*NN_RESIZE), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Original", red_img)
cv2.imwrite(f"{DIRECTORY}/2 original.jpg", red_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
