# signal_prediction
#  **Redes Neurais para Identificação e Tradução de Linguagens de Sinais. | Neural Networks for Sign Language Identification and Translation.**

Esse repositório tem como objetivo compartilhar meu projeto de aplicação de Redes Neurais e Visão Computacional para reconhecimento de sinais provenientes de Linguagens de Sinais como ASL (American Sign Language) ou LIBRAS (Linguagem Brasileira de Sinais).
---
**ETAPA 1: Coleta dos Dados | Data Collect/Extraction**

Utilizando OpenCV e Mediapipe (Python), coleta-se imagens dos landmarks (hands e pose) visuais salvando-as em um diretório (jpg) e suas respectivas coordenadas (x, y, z) em outro diretório (pkl). Define-se os sinais (labels) que serão coletados e após a criação da base de dados, estão prontos para treinamento.
---
**ETAPA 2: Treinamento do Modelo | Model Training**

Utilizando Tensorflow.Keras (Python), cria-se um Generator compatível com dois tipos de entradas diferentes [imagem, landmarks] para classificar com os sinais (labels). É utilizado uma Rede Neural Convolucional (CNN - Convolutional Neural Network) para a entrada de imagens e uma Rede Neural Direta (FNN - Feed-Forward Neural Network) para a entrada das coordenadas. Após as duas entradas, concatena-se as duas em uma camada como fechamento da rede. Dessa maneira a Rede Neural é treinada com dois tipos de dados de entradas diferentes e depois com as duas juntas.
---
**ETAPA 3: Teste do Modelo com Video em Tempo-Real | Real-Time Video Testing**

Gera-se o vídeo em tempo-real utilizando uma webcam para testar os reconhecimentos e identificações de sinais, gerando uma mensagem-texto com as previsões de alta confiança.

