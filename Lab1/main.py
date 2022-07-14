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

def show_result(x, y, pred_y,fname):
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
    plt.savefig(fname)

# activation functions
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1.0 - x**2

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
    # print(f'pred ={prediction.transpose()}')
    return np.mean((prediction - ground_truth) ** 2)

def mse_loss_derivative(prediction, ground_truth):
    return 2 * (prediction - ground_truth) / len(prediction)

class Layer:
    def __init__(self, input_num, output_num, activation, optimizer, learning_rate):
        self.weight = np.random.normal(0,1,(input_num +1 , output_num))
        print('weight=',self.weight)
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = 0
        self.n = 0

    def forward_pass(self, inputs):
        forward_value = np.append(inputs, np.ones((inputs.shape[0],1)),axis = 1)
        self.forward_output = None
        if self.activation == 'sigmoid':
            self.forward_output = sigmoid(np.matmul(forward_value, self.weight))
        elif self.activation == 'tanh':
            self.forward_output = tanh(np.matmul(forward_value, self.weight))
        elif self.activation == 'relu':
            self.forward_output = relu(np.matmul(forward_value, self.weight))
        elif self.activation == 'leaky_relu':
            self.forward_output = leaky_relu(np.matmul(forward_value, self.weight))
        else:
            self.forward_output = np.matmul(forward_value,self.weight)

        self.forward_value = forward_value
        return self.forward_output

    def backward_pass(self, inputs):
        self.backward_output = None
        if self.activation == 'sigmoid':
            self.backward_output = np.multiply(derivative_sigmoid(self.forward_output),inputs)
        elif self.activation == 'tanh':
            self.backward_output = np.multiply(derivative_tanh(self.forward_output), inputs)
        elif self.activation == 'relu':
            self.backward_output = np.multiply(derivative_relu(self.forward_output), inputs)
        elif self.activation == 'leaky_relu':
            self.backward_output = np.multiply(derivative_leaky_relu(self.forward_output), inputs)
        else:
            self.backward_output = inputs
        return np.matmul(self.backward_output, self.weight[:-1].transpose())

    def learn(self):
        gradient = np.matmul(self.forward_value.transpose(), self.backward_output)
        beta = 0.9
        epsilon = 10 ** (-8)
        if self.optimizer == 'gd':
            weight_change = -self.learning_rate * gradient
        elif self.optimizer == 'momentum':
            self.momentum = beta * self.momentum - self.learning_rate * gradient
            weight_change = self.momentum
        elif self.optimizer == 'adagrad':
            self.n += np.square(gradient)
            weight_change = -self.learning_rate * np.multiply(1/(np.sqrt(self.n + epsilon)), gradient)
        self.weight += weight_change

class NeuralNetwork:
    def __init__(self, epoch, learning_rate, layers, inputs, hidden_units, activation, optimizer):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.activation = activation
        self.optimizer = optimizer
        self.learn_epoch =[]
        self.learn_loss = []

        self.layers = [Layer(inputs, hidden_units, activation, optimizer, learning_rate)]

        for i in range(layers -1):
            self.layers.append(Layer(hidden_units, hidden_units, activation, optimizer, learning_rate))
        self.layers.append(Layer(hidden_units, 1, 'sigmoid', optimizer, learning_rate))

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

            if epoch %100 == 0:
                # print('inputs=',inputs,'ground_truth=',ground_truth)
                # print('prediction = ',prediction)
                
                print(f'Epoch {epoch} loss : {loss}')
                self.learn_epoch.append(epoch)
                self.learn_loss.append(loss)
            if loss < 0.001:
                break
            
        return np.round(prediction)

    def show_learning_curve(self, fname):
        plt.figure()
        plt.title('Learning curve', fontsize = 18)
        plt.plot(self.learn_epoch, self.learn_loss)
        plt.savefig(fname +'_learning_curve.png')

def main():
    inputs1, label1 = generate_linear()
    inputs2, label2 = generate_XOR_easy()
    network = NeuralNetwork(epoch = 1000000, learning_rate = 0.05, layers = 2, inputs = 2, hidden_units = 4,
                            activation ='leaky_relu',optimizer = 'gd')   
    # prediction = network.train(inputs1, label1)
    # show_result(inputs1, label1, prediction,'linear.png')
    # network.show_learning_curve('linear')
    # print('Accuracy : ', float(np.sum(prediction == label1)) / len(label1))
    prediction = network.train(inputs2, label2)
    show_result(inputs2, label2,prediction,'XOR.png')
    network.show_learning_curve('XOR')
    print('Accuracy : ', float(np.sum(prediction == label2)) / len(label2))

if __name__ == '__main__':
    main()
# x1 = generate_XOR_easy()[0]
# print(np.append(x1,np.ones((x1.shape[0],1)),axis=1))