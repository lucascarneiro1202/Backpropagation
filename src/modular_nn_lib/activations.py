import numpy as np

# Classe base para funções de ativação
class Activation:
    def forward(self, input_data):
        raise NotImplementedError
    
    def backward(self, output_gradient):
        raise NotImplementedError

# Sigmoid
class Sigmoid(Activation):
    def forward(self, input_data):
        # TODO: Calcular a função sigmoid
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, output_gradient):
        # TODO: Calcular a derivada da sigmoid (gradiente * sigmoid * (1 - sigmoid))
        return output_gradient * self.output * (1 - self.output)