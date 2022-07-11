import numpy as np
import matplotlib.pyplot as plt
import copy
def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21,1)

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

# activation functions
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1.0 - y**2

def relu(x):
    return np.maximum(0.0, x)

def derivative_relu(x):
    return np.heaviside(x, 0.0)

def leaky_relu(x):
    alpha = 0.005
    return np.maximum(alpha * x, x)

def derivative_leaky_relu(x):
    alpha = 0.005
    y = copy.deepcopy(x);
    y[y > 0.0] = 1.0
    y[y <= 0.0] = alpha
    return y 

class Layer:
    def __init__(self, input_num, output_num, activation = 'sigmoid', optimizer = 'gd', learning_rate = 0.05):
        self.weight = np.random.normal(0,1,(input_num +1 , output_num))
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward_pass(self, inputs):
        forward_value = np.append(inputs, np.ones((inputs.shape[0],1)),axis = 1)
        self.forward_output = None
        if self.activation == 'sigmoid':
            self.forward_output = sigmoid(np.matmul(forward_value, self.weight))

        return self.forward_output

    def backward_pass(self, inputs):
        self.backward_output = None
        if self.activation == 'sigmoid':
            self.backward_output = np.multiply(derivative_sigmoid(self.forward_output),inputs)

        return np.matmul(self.backward_output, self.weight[:-1].T)

    def learning(self):
        gradient = np.matmul(self.forward_output.T, self.backward_output)
        if self.optimizer == 'gd':
            weight_change = -self.learning_rate * gradient
        self.weight += weight_change


# x1 = generate_XOR_easy()[0]
# print(np.append(x1,np.ones((x1.shape[0],1)),axis=1))