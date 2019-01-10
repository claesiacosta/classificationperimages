import cv2
import numpy as np
# Classificação vetorial de Suporte Linear
from sklearn.svm import SVC
# Classificação por Regressão Vetorial de Suporte
from sklearn.svm import SVR

atletico_n = cv2.imread("atletico.jpg")
flamengo_n = cv2.imread("flamengo.jpg")
corinthians_n = cv2.imread("corinthians.jpg")
palmeiras_n = cv2.imread("palmeiras.jpg")
teste1 = cv2.imread("teste.jpg")

#Mostra o shape da imagem, pixels, cores...
#print(atletico_n.shape)
#print(flamengo_n.shape)
#print(corinthians_n.shape)
#print(palmeiras_n.shape)
#print(teste1.shape)


#Ver a imagem
#cv2.imshow("atletico", atletico)
#cv2.waitKey(0)#aguardar até fechar a janela

#reduzir tamanhos das imagens, normalmente em bancos
#de dados muitos grandes, para treinamento
atletico = cv2.resize(atletico_n, (10,10))
flamengo = cv2.resize(flamengo_n, (10,10))
corinthians = cv2.resize(corinthians_n, (10,10))
palmeiras = cv2.resize(palmeiras_n, (10,10))
teste = cv2.resize(teste1, (10,10))

#Concatenação das matrizes
X = np.concatenate((atletico, corinthians, flamengo, palmeiras), axis=0)

#cria eixo de índices
y = [1,2,3,4]
#converte pra matriz
y = np.array(y)
Y = y.reshape(-1)

X = X.reshape(len(y), -1)

clf_lin = SVC(kernel='linear')
svr_lin = SVR(kernel='linear')

print('Treinamento SVC - classificação linear de suporte linear')

clf_lin.fit(X,Y)

print("Fim SVC")

predicao = clf_lin.predict(teste.reshape(1,-1))

score = clf_lin.score(X,Y)

print(predicao)
print(score)

if predicao == 1:
	resultado = atletico_n
if predicao == 2:
	resultado = corinthians_n
if predicao == 3:
	resultado = flamengo_n
if predicao == 4:
	resultado = palmeiras_n

#cv2.imshow("Resultado", resultado)
#cv2.imshow("Teste", teste1)
#cv2.waitKey(0)


print('Treinamento SVR -  Regressão vetorial de suporte(Regressão por Vetores de Suporte)')

svr_lin.fit(X,Y)

print("Fim SVR")

predicao = svr_lin.predict(teste.reshape(1,-1))

score = svr_lin.score(X,Y)

print(predicao)
print(score)

if predicao == 1:
	resultado = atletico_n
if predicao == 2:
	resultado = corinthians_n
if predicao == 3:
	resultado = flamengo_n
if predicao == 4:
	resultado = palmeiras_n

cv2.imshow("Resultado", resultado)
cv2.imshow("Teste", teste1)
cv2.waitKey(0)