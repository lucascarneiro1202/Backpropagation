import numpy as np

# Classe base para funções de perda
class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        raise NotImplementedError

# Erro Quadrático Médio (MSE)
class MSE(Loss):
    def forward(self, y_pred, y_true):
        # TODO: Calcular o MSE
        # Dica: np.mean(np.power(y_true - y_pred, 2))
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_pred, y_true):
        # TODO: Calcular a derivada do MSE
        # Dica: 2 * (y_pred - y_true) / np.size(y_true)
        return 2 * (y_pred - y_true) / np.size(y_true)