#Aluna: Viviane Ruotolo
#RA: 2324822

import cv2
import numpy as np

#SMAA: Subpixel Morphological Anti-Aliasing

#Preciso implementar o Canny?
#Borrar a máscara ou a imagem original?
#Substituir diretamente ou usar pesos?

#Filtro de Sobel -> Não usa orientação do gradiente
#Recebo o gradiente e orientação, com a orientação eu descubro pra onde eu vou BORRAR
# Borro em direções selecionadas, não em toda parte que tem borda
# Descubro todas as bordas com o Sobel, borrar direcionado uso a orientação
#  
#Canny -> Elimina pontos que não são máximos locais na direção do gradiente

#Fazer uma máscara para criar a imagem apenas com a bordas

#Aplica suavização na máscara ou imagem

#Substitui nos pixels de borda, a suavização da imagem, nos demais é a img original
#Borrar só o L do hsl
#Aplicar para os três canais da imagem colorida
img = cv2.imread("images/mario.png")
if img is None:
    print ('Error opening image: ')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Constantes arbitrárias?
edges = cv2.Canny(gray, 150, 200)
cv2.imshow("Edges", edges)

b, g, r = cv2.split(img)

#Solução 1: borra a imagem, substitui direto com condição, faz independente nos três canais e junta
img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
b_blur = cv2.GaussianBlur(b, (3, 3), 0)
g_blur = cv2.GaussianBlur(g, (3, 3), 0)
r_blur = cv2.GaussianBlur(r, (3, 3), 0)

b_final = np.where(edges == 255, b_blur, b).astype(np.uint8)
g_final = np.where(edges == 255, g_blur, g).astype(np.uint8)
r_final = np.where(edges == 255, r_blur, r).astype(np.uint8)

img_final = cv2.merge([b_final, g_final, r_final])
cv2.imshow("Anti Aliasing", img_final)
cv2.imwrite("AntiAliasing_gray.jpg", img_final)

#Solução 2: Borra a máscara, mistura com pesos e usa cv2.Color pra ficar colorido
edges_b = cv2.Canny(b, 150, 200)
edges_g = cv2.Canny(g, 150, 200)
edges_r = cv2.Canny(r, 150, 200)

edges_b_blur = cv2.GaussianBlur(edges_b, (3, 3), 0)
edges_g_blur = cv2.GaussianBlur(edges_g, (3, 3), 0)
edges_r_blur = cv2.GaussianBlur(edges_r, (3, 3), 0)

edges_color_3 = cv2.merge([edges_b_blur, edges_g_blur, edges_r_blur])
img_final3 = cv2.addWeighted(img, 0.9, edges_color_3, 0.1, 0)
cv2.imshow("3 AA - Mascara", img_final3)
cv2.imwrite("3 AA - Mascara.jpg", img_final3)


edges_blur = cv2.GaussianBlur(edges, (3, 3), 1)
edges_blur_color = cv2.cvtColor(edges_blur, cv2.COLOR_GRAY2BGR)
img_final2 = cv2.addWeighted(img, 0.9, edges_blur_color, 0.1, 0)

cv2.imshow("Edges - blur", edges_blur)
cv2.imshow("Edges blur color", edges_blur_color)
cv2.imshow("AA - Mascara", img_final2)
cv2.imwrite("AA - Mascara.jpg", img_final2)




cv2.imshow("Original", img)
cv2.imwrite("Original_gray.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()