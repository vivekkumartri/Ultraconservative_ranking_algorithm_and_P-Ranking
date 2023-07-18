import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
def widrowhoff(train,y,rounds,c,epochs):
    """
    Implementation of the Widrow-Hoff algorithm for online learning.

    Parameters:
    - train (numpy.ndarray): The training data as a 2D array.
    - y (list or numpy.ndarray): The target labels for training.
    - rounds (int): The number of rounds or iterations for training.
    - c (float): The learning rate or step size for the algorithm.
    - epochs (int): The number of epochs or iterations for each round.

    Returns:
    - loss (list): The list of loss values at each round.
    - w (numpy.ndarray): The learned weight vector.

    Description:
    The function implements the Widrow-Hoff algorithm for online learning. It takes the training data
    `train`, the target labels `y`, the number of rounds `rounds`, the learning rate `c`, and the number
    of epochs `epochs` as input.

    The algorithm initializes the weight vector `w` with zeros. It then performs label conversion to ensure
    that the labels are in the range 1 to k, where k is the number of unique labels in `y`. 

    The function iterates `epochs` times within each round, and for each iteration, it iterates from 1 to
    `rounds`. It calculates the predicted label `y_bart` using the current weight vector `w` and computes
    the loss `l` as the average absolute difference between the predicted label `y_bart` and the true label `y`.

    During each iteration, the weight vector `w` is updated using the Widrow-Hoff update rule, which adjusts
    the weights based on the error between the predicted and true labels.

    The function returns the list of loss values `loss` at each round and the final learned weight vector `w`.
    """
    w=np.array([0 for i in range(np.shape(train)[1])])
    loss=[]
    label=range(1,len(np.unique(y))+1)
    l2=np.sort(np.unique(y))
    for i in range(len(y)):
        for j in range(len(label)):
            if y[i]==l2[j]:
                y[i]=label[j]
    for s in range(epochs):
        for i in range(1,rounds+1):
            x=np.array(train[i-1])
            y_bart=safe_sparse_dot(w,x)
            y_bar=[]
            for t in range(1,i+1):
                x=np.array(train[t-1])
                y_bart=safe_sparse_dot(w,x)
                y_bar.append(y_bart)
            l=sum([np.absolute(y[j-1]-y_bar[j-1]) for j in range(1,i+1)])/i
            print(f"loss in round {i} is ",l)
            loss.append(l)
            w=np.subtract(w,c*(safe_sparse_dot(w,x)-y[i-1])*x)
    return loss,w


def pranking(train,y,rounds,epochs):
    
    """
    Implementation of the P-Ranking algorithm for ordinal regression.

    Parameters:
    - train (numpy.ndarray): The training data as a 2D array.
    - y (list or numpy.ndarray): The target labels for training.
    - rounds (int): The number of rounds or iterations for training.
    - epochs (int): The number of epochs or iterations for each round.

    Returns:
    - loss (list): The list of loss values at each round.
    - w (numpy.ndarray): The learned weight vector.
    - b (numpy.ndarray): The threshold values for each rank.

    Description:
    The function implements the P-Ranking algorithm for ordinal regression. It takes the training data `train`,
    the target labels `y`, the number of rounds `rounds`, and the number of epochs `epochs` as input.

    The algorithm initializes the weight vector `w` with zeros and the threshold values `b` based on the unique
    labels in `y`. It performs label conversion to ensure that the labels are in the range 1 to k, where k is
    the number of unique labels in `y`.

    The function defines the `predict` function, which predicts the rank based on the current weight vector `w`
    and threshold values `b`.

    The function iterates `epochs` times within each round, and for each iteration, it iterates from 1 to `rounds`.
    It predicts the rank `y_bar` for the input `x` using the `predict` function. If `y_bar` is not equal to the true
    rank `y[i-1]`, it updates the weight vector `w` and threshold values `b` using the P-Ranking update rule.

    At every iteration, the function calculates the loss `l` as the mean absolute difference between the true ranks
    `y[:i]` and the predicted ranks `y_bar1` for all instances up to the current iteration.

    The function returns the list of loss values `loss` at each round, the final learned weight vector `w`, and the
    threshold values `b`.
    """
    w=np.zeros(train.shape[1])
    b = np.zeros(len(np.unique(y)) - 1)
    b = np.append(b, np.inf)
    loss=[]
    label=range(1,len(np.unique(y))+1)
    unique_y = np.sort(np.unique(y))
    y = np.array([np.where(unique_y == labels)[0][0] + 1 for labels in y]) 
    def predict(x, w, b):
        scores = safe_sparse_dot(x, w) < b
        return np.argmax(scores) + 1
    for s in range(epochs):
        for i in range(1,rounds+1):
            x=train[i-1]
            y_bar=predict(x, w, b)
            if y_bar !=y[i-1]:
                y_rt=np.where(y[i-1]<=label[:-1],-1,1)
                t_rt=np.where((safe_sparse_dot(x,w)-b[:-1])*y_rt <=0,y_rt,0)
                w=w+sum(t_rt)*x
                b[:-1] -= t_rt
            #predicting score
            y_bar1=np.array([predict(train[j-1],w,b) for j in range(1,i+1)])
            l = np.mean(np.abs(y[:i] - y_bar1)) #loss function
            print(f"loss in round {i} is ",l)
            loss.append(l)
    return loss,w,b

def uniformmulticlassalgo(feature,y,maxiter,d):
    """
    Implements a multiclass classification algorithm using the Uniform Margin Loss function.

    Args:
        feature (numpy.ndarray): The input features matrix of shape (n_samples, n_features).
        y (numpy.ndarray): The target labels vector of shape (n_samples,).
        maxiter (int): The maximum number of iterations to perform.
        d (int): The number of rounds to run the algorithm.

    Returns:
        tuple: A tuple containing:
            - loss (list): A list of loss values at each round of training.
            - m (numpy.ndarray): The learned weight matrix of shape (n_classes, n_features).

    The algorithm updates the weight matrix 'm' iteratively over 'd' rounds, minimizing the Uniform Margin Loss.
    It converts the original labels 'y' to a zero-based index representation and uses one-hot encoding for multiclass classification.
    The loss value at each round is printed and stored in the 'loss' list.
    The final weight matrix 'm' is returned along with the loss values.
    """
    m=np.zeros((len(np.unique(y)),len(feature[0])))
    loss=[]
    def taugenerate(h,e,yt):
        t=[0 for i in range(h)]
        t[int(yt-1)]=1
        for k in e:
            t[k-1]=-(1/len(e))
        return t
    label=range(1,len(np.unique(y))+1)
    l2=np.sort(np.unique(y))
    for i in range(len(y)):
        for j in range(len(label)):
            if y[i]==l2[j]:
                y[i]=label[j]
    for s in range(d):
        for i in range(1,maxiter+1):
            x=feature[i-1]
            ybart=np.argmax([safe_sparse_dot(m[j],x) for j in range(len(label))])+1
            yt=y[i-1]
            e=[]
            for r in range(1,len(label)+1):
                if  r!=yt:
                    if safe_sparse_dot(m[int(r-1)],x)>=safe_sparse_dot(m[int(yt-1)],x):
                        e.append(r)
            if len(e)!=0:
                tau=taugenerate(len(label),e,yt)
                for k in label:
                    m[k-1]=np.add(m[k-1],tau[k-1]*x)
            y_bar1=[]       
            for t in range(1,i+1):
                x=feature[t-1]
                ybart=np.argmax([safe_sparse_dot(m[j],x) for j in range(len(label))])+1
                y_bar1.append(ybart)
            l=(1/i)*sum([np.absolute(y[j-1]-y_bar1[j-1]) for j in range(1,i+1)])
            print(f"loss in round {i} is ",l)
            loss.append(l)
    return loss,m         

def worstmulticlassalgo(feature,y,maxiter,d):
    """
    Implements a multiclass classification algorithm using the Worst Margin Loss function.

    Args:
        feature (numpy.ndarray): The input features matrix of shape (n_samples, n_features).
        y (numpy.ndarray): The target labels vector of shape (n_samples,).
        maxiter (int): The maximum number of iterations to perform.
        d (int): The number of rounds to run the algorithm.

    Returns:
        tuple: A tuple containing:
            - loss (list): A list of loss values at each round of training.
            - m (numpy.ndarray): The learned weight matrix of shape (n_classes, n_features).

    The algorithm updates the weight matrix 'm' iteratively over 'd' rounds, minimizing the Worst Margin Loss.
    It converts the original labels 'y' to a zero-based index representation and uses one-hot encoding for multiclass classification.
    The loss value at each round is printed and stored in the 'loss' list.
    The final weight matrix 'm' is returned along with the loss values.
    """
    m=np.zeros((len(np.unique(y)),len(feature[0])))
    loss=[]
    label=range(1,len(np.unique(y))+1)
    l2=np.sort(np.unique(y))
    def taugenerate(h,m,x,yt,label):
        t=[0 for i in range(h)]
        t[int(yt-1)]=1
        k=np.argsort([safe_sparse_dot(m[j],x) for j in range(h)])[-2]
        t[k]=-1
        return t
    for i in range(len(y)):
        for j in range(len(label)):
            if y[i]==l2[j]:
                y[i]=label[j]
    for s in range(d):
        for i in range(1,maxiter+1):
            x=feature[i-1]
            ybart=np.argmax([safe_sparse_dot(m[j],x) for j in range(len(label))])+1
            yt=y[i-1]
            e=[]
            for r in range(1,len(label)+1):
                if  r!=yt:
                    if safe_sparse_dot(m[int(r-1)],x)>=safe_sparse_dot(m[int(yt-1)],x):
                        e.append(r)
            if len(e)!=0:
                tau=taugenerate(len(label),m,x,yt,label)
                for k in label:
                    m[k-1]=np.add(m[k-1],tau[k-1]*x)
            y_bar1=[]       
            for t in range(1,i+1):
                x=feature[t-1]
                ybart=np.argmax([safe_sparse_dot(m[j],x) for j in range(len(label))])+1
                y_bar1.append(ybart)
            l=(1/i)*sum([np.absolute(y[j-1]-y_bar1[j-1]) for j in range(1,i+1)])
            print(f"loss in round {i} is ",l)
            loss.append(l)
    return loss,m

def vimulticlassalgo(feature,y,maxiter,d):
    """
    Trains a multi-class classification algorithm using the Variance-based Incremental Multi-class Classification (VI-MC) algorithm.

    Args:
        feature (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Target vector of shape (n_samples,) containing class labels.
        maxiter (int): Maximum number of iterations for training.
        d (int): Number of iterations to calculate the loss.

    Returns:
        tuple: A tuple containing:
            - loss (list): List of loss values calculated at each iteration.
            - m (numpy.ndarray): Trained model matrix of shape (n_classes, n_features).
    """
    m=np.zeros((len(np.unique(y)),len(feature[0])))
    loss=[]
    y_bar1=[]
    label=range(1,len(np.unique(y))+1)
    l2=np.sort(np.unique(y))
    def taugenerate(h,m,x,yt):
        t=[0 for i in range(h)]
        t[int(yt-1)]=1
        for i in range(h):
            if i!=int(yt-1):
                if safe_sparse_dot(m[i],x)-safe_sparse_dot(m[int(yt-1)],x)>0:
                    num=safe_sparse_dot(m[i],x)-safe_sparse_dot(m[int(yt-1)],x)
                else:
                    num=0
                den=0
                for k in range(h):
                    if safe_sparse_dot(m[k],x)-safe_sparse_dot(m[int(yt-1)],x)>0:
                        den+=safe_sparse_dot(m[k],x)-safe_sparse_dot(m[int(yt-1)],x)
                if den==0:
                    den=1
                t[i]=-(num/den)     
        return t
    for i in range(len(y)):
        for j in range(len(label)):
            if y[i]==l2[j]:
                y[i]=label[j]
    for s in range(d):
        for i in range(1,maxiter+1):
            x=feature[i-1]
            ybart=np.argmax([safe_sparse_dot(m[j],x) for j in range(len(label))])+1
            yt=y[i-1]
            e=[]
            for r in range(1,len(label)+1):
                if  r!=yt:
                    if safe_sparse_dot(m[r-1],x)>=safe_sparse_dot(m[int(yt-1)],x):
                        e.append(r)
            if len(e)!=0:
                tau=taugenerate(len(label),m,x,yt)
                for k in label:
                    m[k-1]=np.add(m[k-1],tau[k-1]*x)
            y_bar1=[]       
            for t in range(1,i+1):
                x=feature[t-1]
                ybart=np.argmax([safe_sparse_dot(m[j],x) for j in range(len(label))])+1
                y_bar1.append(ybart)
            l=(1/i)*sum([np.absolute(y[j-1]-y_bar1[j-1]) for j in range(1,i+1)])
            print(f"loss in round {i} is ",l)
            loss.append(l)
    return loss,m

def mira(feature,y,maxiter,d):
    
    """
    Trains a multi-class classification algorithm using the Margin-Infused Relaxed Algorithm (MIRA).

    Args:
        feature (numpy.ndarray): Input feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Target vector of shape (n_samples,) containing class labels.
        maxiter (int): Maximum number of iterations for training.
        d (int): Number of iterations to calculate the loss.

    Returns:
        tuple: A tuple containing:
            - loss (list): List of loss values calculated at each iteration.
            - m (numpy.ndarray): Trained model matrix of shape (n_classes, n_features).
    """
    m=np.ones((len(np.unique(y)),len(feature[0])))
    y_bar1=[]
    loss=[]
    label=range(1,len(np.unique(y))+1)
    l2=np.sort(np.unique(y))
    def taugenerate(h,m,x,yt,ybart):
        t=[0 for i in range(h)]
        t[int(yt-1)]=1
        tau=np.minimum(0.001, (((m[int(ybart-1)]-m[int(yt-1)])*x)+1.0)/(2.0*(safe_sparse_dot(x,x))))
        t[int(yt-1)]=tau
        t[int(ybart-1)]=-tau
        return t
    for i in range(len(y)):
        for j in range(len(label)):
            if y[i]==l2[j]:
                y[i]=label[j]
    for s in range(d):
        for i in range(1,maxiter+1):
            x=feature[i-1]
            ybart=np.argmax([safe_sparse_dot(m[j],x) for j in range(len(label))])+1
            yt=y[i-1]
            tau=taugenerate(len(label),m,x,yt,ybart)
            for k in label:
                m[k-1]=np.add(m[k-1],tau[k-1]*x)
            y_bar1=[]       
            for t in range(1,i+1):
                x=feature[t-1]
                ybart=np.argmax([safe_sparse_dot(m[j],x) for j in range(len(label))])+1
                y_bar1.append(ybart)
            l=(1/i)*sum([np.absolute(y[j-1]-y_bar1[j-1]) for j in range(1,i+1)])
            print(f"loss in round {i} is ",l)
            loss.append(l)
    return loss,m

class OrdinalRegressionNetwork:
    """
    Implementation of an ordinal regression neural network for predicting ordinal categories.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of ordinal categories.

    Methods:
        __init__(self, input_size, hidden_size, output_size):
            Initializes the OrdinalRegressionNetwork with the given input size, hidden size, and output size.

        sigmoid(self, x):
            Applies the sigmoid activation function to the input x.

        forward(self, X):
            Performs the forward pass of the network given the input X and returns the predicted ordinal categories.

        backward(self, X, y_true, y_pred, learning_rate):
            Performs the backward pass of the network and updates the weights and biases based on the given target labels, predicted labels, and learning rate.

        train(self, X, y, learning_rate=0.01, epochs=100):
            Trains the network using the given input features X and target labels y for a specified number of epochs and learning rate. Returns a list of loss values calculated at each epoch.

        predict(self, X):
            Performs the forward pass of the network on the given input features X and returns the predicted ordinal categories.

    Attributes:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of ordinal categories.
        W1 (numpy.ndarray): Weight matrix of shape (input_size, hidden_size) for the connections between the input layer and the hidden layer.
        b1 (numpy.ndarray): Bias matrix of shape (1, hidden_size) for the hidden layer.
        W2 (numpy.ndarray): Weight matrix of shape (hidden_size, output_size) for the connections between the hidden layer and the output layer.
        b2 (numpy.ndarray): Bias matrix of shape (1, output_size) for the output layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.Z1 = safe_sparse_dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = safe_sparse_dot(self.A1, self.W2) + self.b2
        self.O = [self.sigmoid(z) for z in self.Z2]
        return self.O

    def backward(self, X, y_true, y_pred, learning_rate):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # calculate errors and gradients
        delta2 = y_pred - y_true
        dW2 = safe_sparse_dot(self.A1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = safe_sparse_dot(delta2, self.W2.T) * (self.A1 * (1 - self.A1))
        dW1 = safe_sparse_dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        # update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, learning_rate=0.01, epochs=100):
        n_samples = X.shape[0]
        loss=[]

        # encode ordinal categories in target vectors
        y_encoded = np.zeros((n_samples, self.output_size))
        for i, y_i in enumerate(y):
            y_encoded[i, :int(y_i)] = 1

        # train the network
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y_encoded, y_pred, learning_rate)
            y_pred = [np.argmax(o) + 1 for o in y_pred]
            loss.append(np.sum(np.absolute(y_pred-y))/len(y))
        return loss

    def predict(self, X):
        y_pred = self.forward(X)
        y_pred = [np.argmax(o) + 1 for o in y_pred]
        return y_pred
    
def pranking_updated(train,y,t,m):
    """
    Implements the Pranking algorithm updated for ranking ordinal categories.

    Args:
        train (numpy.ndarray): Training feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Target vector of shape (n_samples,) containing ordinal category labels.
        t (int): Maximum number of iterations for training.
        m (int): Number of iterations to calculate the loss.

    Returns:
        tuple: A tuple containing:
            - loss (list): List of loss values calculated at each iteration.
            - w (numpy.ndarray): Weight vector of shape (n_features,) learned during training.
            - b (numpy.ndarray): Bias vector of shape (n_classes - 1,) learned during training.
    """
    w=np.zeros(train.shape[1])
    b = np.zeros(len(np.unique(y)) - 1)
    b = np.append(b, np.inf)
    loss=[]
    label=range(1,len(np.unique(y))+1)
    unique_y = np.sort(np.unique(y))
    y = np.array([np.where(unique_y == labels)[0][0] + 1 for labels in y]) 
    def predict(x, w, b):
        scores = np.absolute(safe_sparse_dot(x, w)-b)
        return np.argmin(scores) + 1
    for s in range(m):
        for i in range(1,t+1):
            x=train[i-1]
            y_bar=predict(x, w, b)
            if y_bar !=y[i-1]:
                y_rt=np.where(y[i-1]<=label[:-1],-1,1)
                t_rt=np.where((safe_sparse_dot(x,w)-b[:-1])*y_rt <=0,y_rt,0)
                w=w+sum(t_rt)*x
                b[:-1] -= t_rt
            #predicting score
              y_bar1=np.array([predict(train[j-1],w,b) for j in range(1,i+1)])
              l = np.mean(np.abs(y[:i] - y_bar1)) #loss function
              print(f"loss in round {i} is ",l)
              loss.append(l)
    return loss,w,b