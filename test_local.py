import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Importa as classes da SUA biblioteca
from modular_nn_lib import NeuralNetwork, DenseLayer, Sigmoid, MSE, SGD

print("Iniciando teste local da biblioteca neural...")

# 1. Gerar dados de teste (problema "duas luas")
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
y = y.reshape(-1, 1) # Ajusta o formato do y para (n_samples, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Construir a Rede Neural
# (Vamos criar uma rede com 2 camadas escondidas)
nn = NeuralNetwork(loss=MSE(), optimizer=SGD(learning_rate=0.01))

nn.add(DenseLayer(input_size=2, output_size=5, activation=Sigmoid()))
nn.add(DenseLayer(input_size=5, output_size=5, activation=Sigmoid()))
nn.add(DenseLayer(input_size=5, output_size=1, activation=Sigmoid()))

# 3. Treinar a Rede
print("Treinando a rede...")
history = nn.train(X_train, y_train, epochs=1000)

# 4. Avaliar e Visualizar
print("Treinamento concluído.")

# Plotar o histórico da perda
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.title('Histórico de Perda (Loss) durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.grid(True)
plt.savefig('local_loss_history.png')
print("Gráfico do histórico de perda salvo como 'local_loss_history.png'")