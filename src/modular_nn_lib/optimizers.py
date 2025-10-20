import numpy as np

# Classe base do otimizador
class Optimizer:
    def update_params(self, layer):
        raise NotImplementedError

# Stochastic Gradient Descent (o clássico)
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        # TODO: Implementar a atualização de pesos do GD
        layer.weights -= self.learning_rate * layer.grad_weights
        layer.bias -= self.learning_rate * layer.grad_bias