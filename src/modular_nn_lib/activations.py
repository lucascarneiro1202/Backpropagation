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
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)
    
class ReLU(Activation):
    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, self.input)
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient * (self.input > 0)