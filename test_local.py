import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from modular_nn_lib import NeuralNetwork, DenseLayer, Sigmoid, ReLU, MSE, SGD

print("Iniciando teste local da biblioteca neural...")

# 1. Gerar dados de teste
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================================================
# EXEMPLO 1: BGD (Batch Gradient Descent) - O Padrão
# ======================================================================
print("\n--- Treinando com BATCH Gradient Descent (Padrão) ---")
nn_bgd = NeuralNetwork(loss=MSE(), optimizer=SGD(learning_rate=1.0))
nn_bgd.add(DenseLayer(2, 5, ReLU()))
nn_bgd.add(DenseLayer(5, 5, ReLU()))
nn_bgd.add(DenseLayer(5, 1, Sigmoid()))

# Chamada padrão: batch_size=None (ou omitido)
history_bgd = nn_bgd.train(X_train, y_train, epochs=1000) 

# ======================================================================
# EXEMPLO 2: SGD (Stochastic Gradient Descent)
# ======================================================================
print("\n--- Treinando com STOCHASTIC Gradient Descent ---")
nn_sgd = NeuralNetwork(loss=MSE(), optimizer=SGD(learning_rate=0.01))
nn_sgd.add(DenseLayer(2, 5, ReLU()))
nn_sgd.add(DenseLayer(5, 5, ReLU()))
nn_sgd.add(DenseLayer(5, 1, Sigmoid()))

# Chamada com batch_size=1
history_sgd = nn_sgd.train(X_train, y_train, epochs=1000, batch_size=1) 

# ======================================================================
# EXEMPLO 3: MBGD (Mini-Batch Gradient Descent)
# ======================================================================
print("\n--- Treinando com MINI-BATCH Gradient Descent ---")
nn_mbgd = NeuralNetwork(loss=MSE(), optimizer=SGD(learning_rate=0.1))
nn_mbgd.add(DenseLayer(2, 5, ReLU()))
nn_mbgd.add(DenseLayer(5, 5, ReLU()))
nn_mbgd.add(DenseLayer(5, 1, Sigmoid()))

# Chamada com batch_size=32 (um valor comum)
history_mbgd = nn_mbgd.train(X_train, y_train, epochs=1000, batch_size=32) 

# ======================================================================
# Visualização Comparativa
# ======================================================================
print("\nGerando gráfico comparativo de perda...")
plt.figure(figsize=(12, 7))
plt.plot(history_bgd, label='BGD (Padrão)', linestyle='-', linewidth=2)
plt.plot(history_sgd, label='SGD (batch_size=1)', linestyle='--', alpha=0.7)
plt.plot(history_mbgd, label='MBGD (batch_size=32)', linestyle=':', linewidth=2)
plt.title('Comparação do Histórico de Perda (Loss)')
plt.xlabel('Época')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.legend()
plt.grid(True)
#plt.ylim(0, 0.25) # Limita o eixo Y para melhor visualização
plt.savefig('local_loss_comparison.png')
print("Gráfico salvo como 'local_loss_comparison.png'")