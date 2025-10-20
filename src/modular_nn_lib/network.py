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
        history = [] # Para guardar o histórico da perda
        
        for epoch in range(epochs):
            # 1. Forward Pass
            y_pred = self.predict(X_train)
            
            # 2. Calcular a Perda (Loss)
            loss = self.loss_func.forward(y_pred, y_train)
            history.append(loss)
            
            # 3. Backward Pass (Backpropagation)
            
            # TODO: Calcular o gradiente inicial (derivada da perda)
            gradient = self.loss_func.backward(y_pred, y_train)
            
            # TODO: Propagar o gradiente para trás pelas camadas
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient, self.optimizer)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
                
        return history