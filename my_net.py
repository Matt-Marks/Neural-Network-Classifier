import numpy as np
from sklearn.model_selection import train_test_split





# The multiplier for the magnitude of the gradient in each gradient decent step.
learning_rate = 0.005

# I use a massive number of epochs. This is because I use 'early stopping' to
# return the network where the accuracy was the highest.
epochs = 3000

# The hidden layers inside the NN.
hidden_layers = [15,15]





def activation(z):
    '''
    Sigmoid function on a vector this is included for use as your
    activation function

    Parameters:
    - z: a vector of elements to preform sigmoid on.

    Output:
    - a vector of elements with sigmoid preformed on them
    '''
    return 1 / (1 + np.exp(-z))

def sigderiv(z):
    '''
    The derivative of Sigmoid, you will need this to preform back prop

    Parameters:
    - z: a vector of elements to preform take the derivative of.

    Output:
    - a vector of elements post derivative.
    '''
    return activation(z) * (1 - activation(z))





class NeuralNetwork(object):

    '''
    An Object that represents a basic neural network. Inside are methods needed
    for it to function.
    '''

    def __init__(self, size, seed=42):
        '''
        The initializarion method for the network.
        The weights and biases will be instantiated to random values.
        This network will change these values by training.

        Parameters:
        - size: A 1D array indicating the node size of each layer
        - seed: An int that will be used to reduce randomness.
        '''

        # Setting random seed.
        self.seed          = seed
        np.random.seed(self.seed)

        # Settings network morphology.
        self.size          = size
        self.weights       = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
        self.biases        = [np.random.rand(n, 1) for n in self.size[1:]]

        # Extra variabels used for early stopping.
        # (Early stopping works by saving the best accuracy over all the epochs,
        # and returning the model that created that given accuracy.)
        self.best_accuracy = 0.0
        self.saved_model   = None

    def restore_best_model(self):
        '''
        Used in early stopping. Once called, this object becomes a copy of the
        saved model.
        '''
        self.seed          = self.saved_model.seed
        self.size          = self.saved_model.size
        self.weights       = self.saved_model.weights
        self.biases        = self.saved_model.biases
        self.best_accuracy = self.saved_model.best_accuracy

    def forward(self, input):
        '''
        Perform a feed forward computation/

        Parameters:
        - input: The data to be fed to the network.

        Returns:
        * A tuple containing:
            - The output value(s) of each example as ‘a’.
            - The values before activation was applied after the input was weighted as ‘pre_activations’.
            - The values after activation for all layers as ‘activations’.
        '''
        a = input
        pre_activations = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
            pre_activations.append(z)
            activations.append(a)
        return a, pre_activations, activations

    def accuracy(self, predicted, correct):
        '''
        A funciton that is used every epoch to compute the accuracy of
        the network.

        Parameters:
        - predicted: A vector containing the predicted output.
        - correct: A vector containing the actual output.

        Returns:
        - A number between 0.0 and 1.0 representing the accuracy.
        '''
        return sum(1 for p,q in zip(predicted,correct) if p == q) / float(len(predicted))

    def loss(self, predicted, y):
        '''
        Returns:
        - The sum of the squared errors of our prediction.
        '''
        return ((y - predicted) ** 2).sum()

    def calcDeltas(self, y, pre_activations, activations):
        '''
        Used in back propagation. This calculates the deltas for each layer.

        Parameters:
        - y: A vector containing the output the network produced.
        - pre_activations: A vector containing the output from each layer prior to activation.
        - activations: A vector containing the output from each layer post activation.

        Returns:
        - A list of lists representing the deltas for each layer of the network.
        '''

        # We start with an empty list.
        deltas = []

        # We calculate the delta for the output layer first.
        output_layer_delta = 2 * (y - activations[-1]) * sigderiv(pre_activations[-1])
        deltas.insert(0, output_layer_delta)

        # We also prepare these reversed lists to make our for-loop indexing easier.
        reversed_weights = self.weights[1:][::-1]
        reversed_pre_activations = pre_activations[::-1][1:]

        # We then loop through each layer of the network in reverse order.
        for (i, w) in enumerate(reversed_weights):

            # We the calculate the delta for that layer and prepend it to the deltas list.
            g = sigderiv(reversed_pre_activations[i])
            delta = np.dot(w.T, deltas[0]) * g
            deltas.insert(0, delta)

        return deltas

    def backpropagate(self, y, pre_activations, activations):
        '''
        Performs back propagation by modifying the weights.

        Parameters:
        - y: A vector containing the output the network produced.
        - pre_activations: A vector containing the output from each layer prior to activation.
        - activations: A vector containing the output from each layer post activation.
        '''

        # We call the delta function and ge the deltas for each layer.
        deltas = self.calcDeltas(y, pre_activations, activations)

        # Then, we calculate new weights for each layer.
        new_weights = []
        for (i, w) in enumerate(self.weights):
            alpha = learning_rate
            new_weights.append(self.weights[i] + alpha * np.dot(activations[i], deltas[i].T).T)

        # Finally, we set the current weights to the modified weights.
        self.weights = new_weights


    def train(self, X, y):
        '''
        Trains the network by feeding the data through it,
        performing backpropagation, and modifying the weights.

        Parameters:
        - X: The featrue set of the network.
        - y: The labels for each fature of the network.
        '''

        # We split the data into training and testing data.
        # This allows us to perform 'early stopping' because
        # we need a bit of testing data to check the accuracy of the network.
        X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2)
        X_train = X_train.T
        X_test  = X_test.T
        y_train = y_train.T
        y_test  = y_test.T

        # We flow data forward and backpropagate for each epoch.
        for _ in range(epochs):

            # Feed data forward.
            a, pre_activations, activations = self.forward(X_train)

            # Backpropagate
            self.backpropagate(y_train, pre_activations, activations)

            # Check the accuracy and update the best model. (early stopping)
            current_accuracy = self.accuracy(self.predict(X_test)[0], y_test[0])
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.saved_model = self

        # Restore the best model found from 'early stopping'.
        self.restore_best_model()





    def predict(self, a):
        '''
        This method will test a vector of input parameter vectors of
        the same form as X in test_train and return the results (Zero or One)
        that your trained network came up with for every element.

        This method does this the same way the included forward method moves an
        input through the network but without storying the previous values
        (which forward stores for use with the delta function you must write)

        Parameters:
        - a: A list of list of input vectors to be tested.
        '''

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        predictions = (a > 0.5).astype(int)
        return predictions





def test_train(X, y):
    '''
    Trains the network.

    It must then train the network given the passed data, where x is
    the parameters in form:

        [[1rst parameter], [2nd parameter], [nth parameter]]

        (Where if there are 100 training examples each of the n lists
        inside the list above will have 100 elements)

    Y is the target which is guarenteed to be binary, or in other
    words true or false. Y will be of the form:

        [[1, 0, 0, ...., 1, 0, 1]]

        (where 1 indicates true and zero indicates false)

    '''

    inputSize = np.size(X, 0)
    layers = hidden_layers.copy()
    layers.insert(0, inputSize)
    layers.append(1)
    retNN = NeuralNetwork(layers)

    #train your network here
    retNN.train(X, y)

    #then the function MUST return your TRAINED nueral network
    return retNN
