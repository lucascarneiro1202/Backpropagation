import numpy as np

# A classe principal que gerencia a rede
class NeuralNetwork:
    def __init__(self, loss: 'Loss', optimizer: 'Optimizer'):
        self.layers = []
        self.loss_func = loss
        self.optimizer = optimizer
    
    def add(self, layer: 'Layer'):
        self.layers.append(layer)
    
    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X_train, y_train, epochs):
        history = [] # Para guardar o histórico da perda média da época
        n_samples = len(X_train)
        
        for epoch in range(epochs):
            # Acumulador para a perda da época
            epoch_loss = 0

            # Embaralha os dados e rótulos juntos
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            # Loop interno que itera sobre cada amostra individual
            for i in range(n_samples):                
                # Pega a amostra 'i'
                # Usamos slicing [i:i+1] para manter o formato 2D (ex: (1, 2))
                x_i = X_train[i:i+1]
                y_i = y_train[i:i+1]
                
                # 1. Forward Pass (para UMA amostra)
                y_pred_i = self.predict(x_i)
                
                # 2. Calcular a Perda (Loss) (para UMA amostra)
                loss = self.loss_func.forward(y_pred_i, y_i)
                epoch_loss += loss
                
                # 3. Backward Pass (Backpropagation) (para UMA amostra)
                
                # Gradiente inicial da perda (para UMA amostra)
                gradient = self.loss_func.backward(y_pred_i, y_i)
                
                # Propaga o gradiente para trás
                # O otimizador será chamado DENTRO de cada 'layer.backward'
                # e os pesos serão atualizados a cada chamada.
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.optimizer)
            
            # Calcula a perda média da época e armazena no histórico
            average_epoch_loss = epoch_loss / n_samples
            history.append(average_epoch_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {average_epoch_loss:.4f}')
                
        return history