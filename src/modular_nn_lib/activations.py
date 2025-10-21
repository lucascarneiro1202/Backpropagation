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
    
class ReLU(Activation):
    def forward(self, input_data):
        # A ReLU é simplesmente max(0, x)
        self.input = input_data
        self.output = np.maximum(0, self.input)
        return self.output
    
    def backward(self, output_gradient):
        # A derivada da ReLU é 1 se x > 0, e 0 caso contrário.
        # Multiplicamos o gradiente de entrada por essa derivada.
        return output_gradient * (self.input > 0)