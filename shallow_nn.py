# helper functions
def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1.0/(1.0+np.exp(-x))
    return s


# main shallow neural network class
class shallowNN(object):

    def __init__(self, X, Y, num_iterations=1000, learning_rate=1.0, weight_scalar=1.0, num_hiddenunits=4, print_cost=False):

        """
        Argument:
        X                   -- input data of size (number of features, number of examples)
        Y                   -- input label data of size (1, number of examples)
        num_iterations      -- number of iterations in gradient descent loop
        learning_rate       -- network learning rate 
        weight_init_scalar  -- weight initialization scalar (weights * scalar)
        num_hiddenunits     -- number of hidden units in the first layer
        print_cost          -- print cost every iteration of gradient descent
        """
        
        # store a reference to the input datasets
        self.X = X
        self.Y = Y
 
        # compute layer sizes given provided dataset
        n_x, n_h, n_y = self.layer_sizes(X, Y, num_hiddenunits)

        # store layer sizes    
        self.n_x = n_x # input layer
        self.n_h = n_h # hidden layer
        self.n_y = n_y # output layer

        # store learning rate, number of iterations
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        # initialize parameters (weights and biases)
        self.parameters = self.initialize_parameters(n_x, n_h, n_y, weight_scalar)
        
        # create neural net model

        self.nn_model(print_cost)
        
    def initialize_parameters(self, n_x, n_h, n_y, weight_scalar):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        params -- python dictionary containing your parameters:
                W1 -- weight matrix of shape (n_h, n_x)
                b1 -- bias vector of shape (n_h, 1)
                W2 -- weight matrix of shape (n_y, n_h)
                b2 -- bias vector of shape (n_y, 1)
        """

        #setup random seed 
        np.random.seed(2)         

        # initialize weights and biases for the hidden and output layers
        W1 = np.random.randn(n_h, n_x) * weight_scalar
        b1 = np.zeros((n_h, 1))  * weight_scalar
        W2 = np.random.randn(n_y, n_h) * weight_scalar
        b2 = np.zeros((n_y, 1)) * weight_scalar


        # store parameters within a dictionary for ease of use
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters

        
    def layer_sizes(self, X, Y, num_hiddenunits):
        """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)
        
        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
    
        n_x = X.shape[0] # size of input layer
        n_y = Y.shape[0] # size of output layer
        return (n_x, num_hiddenunits, n_y)

    def forward_propagation(self, X, parameters):
        """
        Argument:
        X          -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)
        
        Returns:
        A2    -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        
        # Forward Propagate to calculate A2 (probabilities)
        Z1 = np.dot(W1,X) + b1
        A1 = np.tanh(Z1) # use tanh activation function for the hidden layer
        Z2 = np.dot(W2,A1) + b2
        A2 = sigmoid(Z2) # use sigmoid activation function for the output layer 


        # store cache of the probabilities inside a dictionary
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
        
        return A2, cache


    def compute_cost(self, A2, Y, parameters):
        """
        Computes the cross-entropy cost 
        
        Arguments:
        A2         -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y          -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing parameters W1, b1, W2 and b2

        Returns:
        cost -- cross-entropy cost given equation (13)
        """
        
        m = Y.shape[1] # number of example

        # Compute the cross-entropy cost
        logprobs = np.dot(Y, np.log(A2).T) + np.dot(1.0 - Y, np.log(1.0 - A2).T)  
        cost = -np.sum(logprobs, axis = 1, keepdims = True)/m

        
        cost = float(np.squeeze(cost))  # squeeze cost from an array into a regular float number
                                        # E.g., turns [[17]] into 17 

        
        return cost

    def backward_propagation(self, parameters, cache, X, Y):
        """
        Backward propagation implementation 
        
        Arguments:
        parameters -- python dictionary containing our parameters 
        cache      -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X          -- input data of shape (2, number of examples)
        Y        -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]
        
        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = parameters["W1"]
        W2 = parameters["W2"]

            
        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache["A1"]
        A2 = cache["A2"]

        
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        dZ2 = A2 - Y
        dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
        db2 = (1.0 / m) * np.sum(dZ2, axis = 1, keepdims = True)
        dZ1 = np.dot(W2.T,  dZ2) * (1.0 - np.power(A1, 2))
        dW1 = (1.0 / m) * np.dot(dZ1, X.T)
        db1 = (1.0 / m) * np.sum(dZ1, axis = 1, keepdims = True)
        
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        
        return grads


    
    def update_parameters(self, parameters, grads):
        """
        Updates parameters using the gradient descent update
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads      -- python dictionary containing your gradients 
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        ### END CODE HERE ###
        
        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        
        # Update rule for each parameter
        W1 = W1 - dW1 * self.learning_rate
        b1 = b1 - db1 * self.learning_rate
        W2 = W2 - dW2 * self.learning_rate
        b2 = b2 - db2 * self.learning_rate

        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        return parameters


    def nn_model(self, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        # retrieve dataset variables, parameters, and anything else that we need
        X, Y = self.X, self.Y
        n_x, n_h, n_y = self.n_x, self.n_h, self.n_y
        
        parameters = self.parameters
   
        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
             
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = self.forward_propagation(X, parameters)
            
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = self.compute_cost(A2, Y, parameters)
     
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = self.backward_propagation(parameters, cache, X, Y)
             
            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = self.update_parameters(parameters, grads)
            
            ### END CODE HERE ###
            
            # Print the cost every 1000 iterations
            if print_cost and i % 1 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        # update self.parameters
        self.parameters = parameters
        
        return parameters
    
    def predict(self, X):
        """
        Arguments:
        X -- dataset of shape (n_x, number of examples to predict)

        Returns:
        Predicted output given input matrix X 
        """
    
        A2, cache   = self.forward_propagation(X, self.parameters)
        predictions = A2 > 0.5
        return A2
