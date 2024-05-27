# to use this py file as logistic regression do the follwoing:
# 1. upload the file and import it using "import log_reg" in the file where you want to use this LogisticRegression class
# 2. use classifier = log_reg.LogisticRegression(learning_rate = 0.01, no_of_iterations = 1000)

import numpy as np

# creates a logistic regression class so you do not need to implement the logistic regression everytime you are using it!
# The self parameter is a reference to the current instance of the class, and is used to access variables that belongs to the class.
# self is always the first argument to any function of a class! 
class LogisticRegression():
    
    
    # declares learning rate and number of iterations (hyperparameters)
    def __init__(self, learning_rate, no_of_iterations): #initializes the parameters of the LogisticRegression class
        
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
     
    
    # trains the model with dataset; X: feature matrix, and Y:target coloumn    
    def fit(self, X, Y): #fits the train dataset to LogisticRegression model
        
        # here, first thing is to determine the number of number of data points and number of input features in your dataset
        # number of data points in the dataset (number of rows) --> m
        # number of input features in the dataset (number of coloumns) --> n
        self.m, self.n = X.shape
        
        # m is needed to find the derivatives and n is needed to find the size of weight matrix
        # initilizes weight and bias values
        self.w = np.zeros(self.n) #numpy array with all the weight values related to each feature is set as 0
        self.b = 0
        
        self.X = X
        self.Y = Y
        
        # implementing gradient descent for optimization
        # creates the gradient descent algorithm to update the weight and bias value
        for i in range(self.no_of_iterations): #instead of no_of_iterations self.no_of_iterations made it successful!
            self.update_weights()      
    
    # 
    def update_weights(self, ): #updates the weight and bias to get get the optimal model
        
        # Y_hat value(sigmoid function)
        # Y_hat = 1 / (1 + np.exp(-z))
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b))) #dot represents matrix multiplication
        
        # derivatives
        dw = (1/self.m)*np.dot((self.X.T), (Y_hat - self.Y)) #T represnts the transpose of X is taken to match the matrix multuplication dimension rule
        db = (1/self.m)*np.sum(Y_hat - self.Y)
        
        # update the weight and bias using gradient descent
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        
    
    # write sigmoid equation and the decision boundary
    def predict(self, X): #pridcits the y value on the test data after the model is trained
        # if Y_predicted > 0.5 => y =1
        # if Y_predicted < 0.5 => y =0
        Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b))) #here self.X is not required as we are predicting on X
        
        # converts the predicted decimal value to binary value
        Y_pred = np.where(Y_pred > 0.5 , 1, 0) # if Y_pred > 0.5 => 1 else 0
        
        return Y_pred
