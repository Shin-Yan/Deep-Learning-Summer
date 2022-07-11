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

#loss functions
def mse_loss(prediction, ground_truth):
    return np.mean((prediction - ground_truth) ** 2)

def mse_loss_derivative(prediction, ground_truth):
    return 2 * (prediction - ground_truth) / len(prediction)

class Layer:
    def __init__(self, input_num, output_num, activation , optimizer , learning_rate ):
        self.weight = np.random.normal(0,1,(input_num +1 , output_num))
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward_pass(self, inputs):
        forward_value = np.append(inputs, np.ones((inputs.shape[0],1)),axis = 1)
        self.forward_output = None
        if self.activation == 'sigmoid':
            self.forward_output = sigmoid(np.matmul(forward_value, self.weight))
        self.forward_value = forward_value
        return self.forward_output

    def backward_pass(self, inputs):
        self.backward_output = None
        if self.activation == 'sigmoid':
            self.backward_output = np.multiply(derivative_sigmoid(self.forward_output),inputs)

        return np.matmul(self.backward_output, self.weight[:-1].T)

    def learn(self):
        gradient = np.matmul(self.forward_value.T, self.backward_output)
        if self.optimizer == 'gd':
            weight_change = -self.learning_rate * gradient
        self.weight += weight_change

class NeuralNetwork:
    def __init__(self,epoch, learning_rate, layers, inputs, hidden_units, activation, optimizer):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.activation = activation
        self.optimizer = optimizer

        self.layers = [Layer(inputs, hidden_units, activation, optimizer, learning_rate)]

        for i in range(layers -1):
            self.layers.append(Layer(hidden_units, hidden_units, activation, optimizer, learning_rate))

        self.layers.append(Layer(hidden_units, 1, activation, optimizer, learning_rate))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

    def backward(self, loss_derivative):
        for layer in self.layers[::-1]:
            loss_derivative = layer.backward_pass(loss_derivative)

    def learn(self):
        for layer in self.layers:
            layer.learn()

    def train(self, inputs, ground_truth):
        for epoch in range(self.epoch):
            prediction = self.forward(inputs)
            loss = mse_loss(prediction, ground_truth)
            self.backward(mse_loss_derivative(prediction, ground_truth))
            self.learn()
        
            self.prediction = prediction
            if epoch%100 == 0:
                print(f'Epoch {epoch} loss : {loss}')
            if loss < 0.001:
                break
def main():
    inputs1, label1 = generate_linear()
    inputs2, label2 = generate_XOR_easy()
    
    network = NeuralNetwork(epoch = 1000000, learning_rate = 0.01, layers = 2, inputs = 2, hidden_units = 4,
                            activation = 'sigmoid', optimizer = 'gd')   
    network.train(inputs1, label1)
    show_result(inputs1, label1, network.prediction)
    network.train(inputs2, label2)
    show_result(inputs2, label2,network.prediction)

if __name__ == '__main__':
    main()
# x1 = generate_XOR_easy()[0]
# print(np.append(x1,np.ones((x1.shape[0],1)),axis=1))