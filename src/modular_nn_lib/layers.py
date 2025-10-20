import numpy as np
from .activations import Activation

# Classe base para camadas
class Layer:
    def forward(self, input_data):
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

# Camada Densa (Totalmente Conectada)
class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation: Activation):
        self.activation = activation
        # TODO: Inicializar pesos e bias (ex: com np.random.randn)
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
    
    def forward(self, input_data):
        self.input = input_data
        # TODO: Calcular a saída Z = (X.W + b)
        self.z = np.dot(self.input, self.weights) + self.bias
        # TODO: Aplicar a função de ativação (A = g(Z))
        return self.activation.forward(self.z)

    def backward(self, output_gradient, optimizer):
        # TODO: Aplicar a derivada da função de ativação
        activation_gradient = self.activation.backward(output_gradient)
        
        # TODO: Calcular os gradientes para pesos e bias
        self.grad_weights = np.dot(self.input.T, activation_gradient)
        self.grad_bias = np.sum(activation_gradient, axis=0, keepdims=True)
        
        # TODO: Calcular o gradiente a ser passado para a camada anterior
        input_gradient = np.dot(activation_gradient, self.weights.T)
        
        # TODO: Usar o otimizador para atualizar os pesos
        optimizer.update_params(self)
        
        return input_gradient