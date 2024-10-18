class NeuralNetwork:
    def __init__(self, layers, lambd=0.01):  # lambd is the regularization strength
        self.layers = layers
        self.lambd = lambd
        self.parameters = self.initialize_parameters()
    
    def compute_cost_with_regularization(self, AL, Y):
        m = Y.shape[1]
        cross_entropy_cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        
        # L2 regularization cost
        L2_cost = 0
        for l in range(1, len(self.parameters)//2 + 1):
            L2_cost += np.sum(np.square(self.parameters[f"W{l}"]))
        L2_cost = (self.lambd / (2 * m)) * L2_cost
        
        cost = cross_entropy_cost + L2_cost
        return cost

    def update_parameters_with_regularization(self, grads, learning_rate):
        for l in range(1, len(self.parameters)//2 + 1):
            self.parameters[f"W{l}"] -= learning_rate * (grads[f"dW{l}"] + (self.lambd / grads[f"dW{l}"].shape[1]) * self.parameters[f"W{l}"])
            self.parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
