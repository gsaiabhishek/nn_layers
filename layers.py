'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Compute gradient wrt input to this layer
        # Update parameters
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        if self.activation == 'relu':
            answer = relu_of_X((X@(self.weights)) + (self.biases))
            self.data = answer.copy()
            return answer
            #raise NotImplementedError
        elif self.activation == 'softmax':
            answer = softmax_of_X((X@(self.weights)) + (self.biases))
            self.data = answer.copy()
            return answer
            #raise NotImplementedError

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        pass
        # END TODO      
    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        if self.activation == 'relu':

            temp = gradient_relu_of_X(self.data, delta)
            n = (self.data).shape[0]
            gradient1 = (1/n)*(activation_prev.T @ temp)
            gradient2 = (1/n)*np.sum(temp, axis = 0)

            prev_delta = temp @ ((self.weights).T)
            self.weights = self.weights - lr*gradient1
            self.biases = self.biases - lr*gradient2
            return prev_delta
            #raise NotImplementedError
        elif self.activation == 'softmax':

            temp = gradient_softmax_of_X(self.data, delta)
            n = (self.data).shape[0]
            gradient1 = (1/n)*(activation_prev.T @ temp)
            gradient2 = (1/n)*np.sum(temp, axis = 0)

            prev_delta = temp @ ((self.weights).T)
            self.weights = self.weights - lr*gradient1
            self.biases = self.biases - lr*gradient2
            return prev_delta
            #raise NotImplementedError

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        pass
        # END TODO
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))
        self.biases = np.random.normal(0,0.1,self.out_depth)
        

    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO


        if self.activation == 'relu':
            n = X.shape[0]
            answer = np.zeros([n, self.out_depth, self.out_row, self.out_col])
            for i in range(self.out_row):
                x_initial = i*self.stride
                x_final = x_initial + self.filter_row

                for j in range(self.out_col):
                    y_initial = j*self.stride
                    y_final = y_initial + self.filter_col

                    answer[:, :, i, j] = np.sum(X[:, np.newaxis, :, x_initial:x_final, y_initial:y_final]*self.weights[np.newaxis,:,:,:,:], axis = (2,3,4))
            bias = self.biases[np.newaxis, :, np.newaxis, np.newaxis]
            answer = answer + bias
            answer = relu_of_X(answer)
            self.data = answer.copy()
            return answer
            #raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        # TODO

        ###############################################
        if self.activation == 'relu':
            n = (self.data).shape[0]
            temp = gradient_relu_of_X(self.data, delta)
            prev_delta = np.zeros([n, self.in_depth, self.in_row, self.in_col])
            gradient1 = np.zeros([self.out_depth, self.in_depth, self.filter_row, self.filter_col])
            gradient2 = (1/n)*np.sum(temp, axis = (0,2,3))

            for i in range(self.out_row):
                x_initial = i*self.stride
                x_final = x_initial + self.filter_row

                for j in range(self.out_col):
                    y_initial = j*self.stride
                    y_final = y_initial + self.filter_col

                    prev_delta[:, :, x_initial:x_final, y_initial:y_final] += np.sum(self.weights[np.newaxis,:,:,:,:]*temp[:,:, np.newaxis, i:i+1, j:j+1], axis = 1)
                    gradient1 += np.sum(activation_prev[:, np.newaxis, :, x_initial:x_final, y_initial:y_final]*temp[:, :, np.newaxis, i:i+1, j:j+1], axis = 0)

            gradient1 = gradient1/n
            self.weights = self.weights - lr*gradient1
            self.biases = self.biases - lr*gradient2

            return prev_delta
            # raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

        # END TODO
    
class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        n = X.shape[0]
        answer = np.zeros([n, self.out_depth, self.out_row, self.out_col])
        for i in range(self.out_row):
            x_initial = i*self.stride
            x_final = x_initial + self.filter_row

            for j in range(self.out_col):
                y_initial = j*self.stride
                y_final = y_initial + self.filter_col

                answer[:, :, i, j] = np.sum(X[:, :, x_initial:x_final, y_initial:y_final], axis = (2,3))
        answer = answer/(self.filter_row * self.filter_col)
        self.data = answer.copy()
        return answer
        #pass
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        n = activation_prev.shape[0]
        answer = np.zeros([n, self.in_depth, self.in_row, self.in_col])
        for i in range(self.out_row):
            x_initial = i*self.stride
            x_final = x_initial + self.filter_row

            for j in range(self.out_col):
                y_initial = j*self.stride
                y_final = y_initial + self.filter_col

                answer[:, :, x_initial:x_final, y_initial:y_final] = delta[:, :, np.newaxis, i, np.newaxis, j]
        answer = answer/(self.filter_row * self.filter_col)
        return answer
        #pass
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        n = X.shape[0]
        answer = np.zeros([n, self.out_depth, self.out_row, self.out_col])
        for i in range(self.out_row):
            x_initial = i*self.stride
            x_final = x_initial + self.filter_row

            for j in range(self.out_col):
                y_initial = j*self.stride
                y_final = y_initial + self.filter_col

                answer[:, :, i, j] = np.max(X[:, :, x_initial:x_final, y_initial:y_final], axis = (2,3))
        self.data = answer.copy()
        return answer
        #pass
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        n = activation_prev.shape[0]
        answer = activation_prev.copy()

        for i in range(self.out_row):
            x_initial = i*self.stride
            x_final = x_initial + self.filter_row

            for j in range(self.out_col):
                y_initial = j*self.stride
                y_final = y_initial + self.filter_col

                temp = (answer[:, :, x_initial:x_final, y_initial:y_final] == (np.max(answer[:,:,x_initial:x_final, y_initial:y_final], axis = (2,3))[:,:,np.newaxis,np.newaxis]) )
                answer[:, :, x_initial:x_final, y_initial:y_final] = answer[:, :, x_initial:x_final, y_initial:y_final]*temp
                answer[:, :, x_initial:x_final, y_initial:y_final] = (answer[:, :, x_initial:x_final, y_initial:y_final] != 0)
                answer[:, :, x_initial:x_final, y_initial:y_final] = answer[:, :, x_initial:x_final, y_initial:y_final]*delta[:,:,np.newaxis,i,np.newaxis,j]
        return answer

        #pass
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # TODO
        # print(X.shape)
        self.n, self.depth, self.rows, self.columns = X.shape
        X = X.reshape(self.n, (self.depth)*(self.rows)*(self.columns))
        return X
        #pass
    def backwardpass(self, lr, activation_prev, delta):
        X = delta.copy()
        X = X.reshape(self.n, self.depth, self.rows, self.columns)
        return X
        #pass
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    Y = X.copy()
    Y[Y < 0] = 0
    return Y
    #raise NotImplementedError
    # END TODO 
    
def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    Y = X.copy()
    Y[Y > 0] = 1.0
    Y[Y < 0] = 0.0
    Y = delta*Y
    return Y
    #raise NotImplementedError
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    Y = X.copy()
    r,c = Y.shape
    Z = np.zeros(r)

    Y = np.exp(Y)
    Z = np.sum(Y, axis = 1)
    Z = Z[:, np.newaxis]
    Y = Y/Z
    return Y
    # return 1/(1+np.exp(-X))
    #raise NotImplementedError
    # END TODO  
def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO
    Y = X.copy()
    r,c = delta.shape
    T1 = np.zeros([r,c])
    T2 = np.zeros([r,c])
    T3 = np.zeros([r,c])

    T1 = Y*(1-Y)
    T1 = T1*delta

    T2 = delta*Y
    T2 = np.sum(T2, axis = 1)
    T2 = np.repeat(T2[:, np.newaxis], c, axis = 1)
    T2 = T2*Y

    T3 = Y*Y
    T3 = T3*delta

    T1 = T1 - T2
    T1 = T1 + T3
    return T1
    #raise NotImplementedError
    # END TODO
