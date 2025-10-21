import numpy as np

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

    # ======================================================================
    # MÉTODO TRAIN MODIFICADO PARA ACEITAR batch_size
    # ======================================================================
    def train(self, X_train, y_train, epochs, batch_size=None):
        """
        Treina a rede neural.
        
        Parâmetros:
        - X_train, y_train: Dados de treinamento
        - epochs: Número de épocas
        - batch_size: Tamanho do lote.
            - Se None (Padrão): Usa Batch Gradient Descent (batch_size = N)
            - Se 1: Usa Stochastic Gradient Descent
            - Se > 1: Usa Mini-Batch Gradient Descent
        """
        history = []
        n_samples = len(X_train)
        
        # Define o tamanho do batch
        # Se batch_size não for fornecido, o padrão é BGD
        if batch_size is None:
            batch_size = n_samples
            
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Embaralha os dados a cada época (bom para SGD e MBGD)
            permutation = np.random.permutation(n_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            # -----------------------------------------------------------------
            # NOVO: Loop de Mini-Batch
            # Itera sobre os dados em "pedaços" (batches)
            # -----------------------------------------------------------------
            for i in range(0, n_samples, batch_size):
                # Define o início e o fim do batch atual
                end = i + batch_size
                x_batch = X_train_shuffled[i:end]
                y_batch = y_train_shuffled[i:end]

                # 1. Forward Pass (para UM batch)
                y_pred_batch = self.predict(x_batch)
                
                # 2. Calcular a Perda (Loss) (para UM batch)
                # O MSE já é uma média, então ponderamos pelo tamanho do batch
                loss = self.loss_func.forward(y_pred_batch, y_batch)
                epoch_loss += loss * len(x_batch)
                
                # 3. Backward Pass (Backpropagation) (para UM batch)
                
                # Gradiente inicial da perda
                gradient = self.loss_func.backward(y_pred_batch, y_batch)
                
                # Propaga o gradiente e atualiza os pesos
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, self.optimizer)
            
            # Fim do loop dos batches
            # -----------------------------------------------------------------
            
            # Calcula a perda média da época
            average_epoch_loss = epoch_loss / n_samples
            history.append(average_epoch_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {average_epoch_loss:.4f}')
                
        return history