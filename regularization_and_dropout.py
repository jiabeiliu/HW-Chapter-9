class NeuralNetwork:
    def __init__(self, layers, keep_prob=0.8):  # keep_prob is the probability of keeping a neuron active
        self.layers = layers
        self.keep_prob = keep_prob
        self.parameters = self.initialize_parameters()

    def forward_propagation_with_dropout(self, A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b
        
        if activation == "relu":
            A = np.maximum(0, Z)
        elif activation == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
        
        # Apply dropout
        D = np.random.rand(A.shape[0], A.shape[1]) < self.keep_prob
        A = np.multiply(A, D)  # shut down some neurons
        A /= self.keep_prob  # scale the values to maintain output
        
        return A, D

    def backward_propagation_with_dropout(self, dA, D, A_prev, W, keep_prob):
        dA = np.multiply(dA, D)  # only propagate the gradient where D == 1
        dA /= keep_prob
        m = A_prev.shape[1]
        dZ = dA * (A_prev > 0)  # if using ReLU
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db
