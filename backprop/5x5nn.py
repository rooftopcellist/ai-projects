'''
Backpropagation Algorithm Implementation
Author: Christian M. Adams

A backwards propagating neural network to learn how to calculate XOR from training
data.  It will be trained on 10 XOR I/O sets and then tested with at least 121 sets
of inputs in order to provide complete coverage.

Values .1 and .9 were used instead of 0 and 1 because the network cannot learn on 0's.

Periodically evaluate network performance.  STOP training when the TSSE < TSSE threshold set

TESTING
1. Generate Testing Set of 121 sets.  Example: <<.1, .9>, .9>
                                               <<.9, .1>, .9>
                                               <<.9, .9>, .1>

'''

import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D



class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.num_outputs = num_outputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def get_output_layer(self):
        return self.output_layer.get_outputs()

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)


        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):


            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]


            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

    def calculate_rmse(self, training_sets):
        rmse = math.sqrt((2 * self.calculate_total_error(training_sets)/(len(training_sets) * self.num_outputs)))
        return rmse

class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print "Neurons: ", len(self.neurons)
        for n in range(len(self.neurons)):
            print " Neuron ", n
            for w in range(len(self.neurons[n].weights)):
                print "  Input: ", self.neurons[n].inputs
                print "  Input Weight: ", self.neurons[n].weights[w]
            print "  Bias:", self.bias

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))


    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2


    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)


    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)


    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]


training_sets = [
      [[0.9, 0.1, 0.1, 0.1, 0.1,
    	0.1, 0.9, 0.1, 0.1, 0.1,
    	0.1, 0.1, 0.9, 0.1, 0.1,
    	0.1, 0.1, 0.1, 0.9, 0.1,
    	0.1, 0.1, 0.1, 0.1, 0.9]
	,
	[0.9, 0.1, 0.1, 0.1, 0.1, .1]],

[
       [0.1, 0.1, 0.1, 0.1, 0.9,
    	0.1, 0.1, 0.1, 0.9, 0.1,
    	0.1, 0.1, 0.9, 0.1, 0.1,
    	0.1, 0.9, 0.1, 0.1, 0.1,
    	0.9, 0.1, 0.1, 0.1, 0.1]
	,
	[0.1, 0.9, 0.1, 0.1, 0.1, .1]],

[
       [0.1, 0.1, 0.9, 0.1, 0.1,
    	0.1, 0.1, 0.9, 0.1, 0.1,
    	0.9, 0.9, 0.9, 0.9, 0.9,
    	0.1, 0.1, 0.9, 0.1, 0.1,
    	0.1, 0.1, 0.9, 0.1, 0.1]
	,
	[0.1, 0.1, 0.9, 0.1, 0.1, .1]],

[
       [0.1, 0.1, 0.1, 0.1, 0.1,
    	0.1, 0.1, 0.1, 0.1, 0.1,
    	0.9, 0.9, 0.9, 0.9, 0.9,
    	0.1, 0.1, 0.1, 0.1, 0.1,
    	0.1, 0.1, 0.1, 0.1, 0.1]
	,
	[0.1, 0.1, 0.1, 0.9, 0.1, .1]],

[
       [0.1, 0.1, 0.9, 0.1, 0.1,
	0.1, 0.1, 0.9, 0.1, 0.1,
	0.1, 0.1, 0.9, 0.1, 0.1,
	0.1, 0.1, 0.9, 0.1, 0.1,
	0.1, 0.1, 0.9, 0.1, 0.1]
	,
	[0.1, 0.1, 0.1, 0.1, 0.9, .1]],

[
       [0.9, 0.1, 0.1, 0.1, 0.9,
	0.1, 0.9, 0.1, 0.9, 0.1,
	0.1, 0.1, 0.9, 0.1, 0.1,
	0.1, 0.9, 0.1, 0.9, 0.1,
	0.9, 0.1, 0.1, 0.1, 0.9]
	,
	[0.1, 0.1, 0.1, 0.1, 0.1, .9]]
]


'''
    1. \
    2. /
    3. +
    4. -
    5. |
    6. X
'''


# Starts Timer
start_time = time.time()

# num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None
# nn = NeuralNetwork(len(training_sets[0][0]), 4, len(training_sets[0][1]), hidden_layer_bias = .7, output_layer_bias = .7)
nn = NeuralNetwork(len(training_sets[0][0]), 4, len(training_sets[0][1]))

# output_list = []
error_rmse = []
error_list = []
ff_outputs = []
count = 0
while nn.calculate_rmse(training_sets) > .05:
    count += 1
    training_inputs, training_outputs = random.choice(training_sets)
    nn.train(training_inputs, training_outputs)
    error_list.append(nn.calculate_total_error(training_sets))

    # if count % 1000 == 0:
    output_value = nn.feed_forward([0.1, 0.9])
    ff_outputs.append(output_value)
    error_rmse.append(nn.calculate_rmse(training_sets))

nn.inspect()
print "--- %s seconds ---" % (time.time() - start_time)
print "Number of Epochs: ", count


#Generate Test Sets
def gen_test_matrix(training_sets):
	t_matrix = []
	t_input = []
	t_output = []
	for full_set in training_sets:
		t_input.append(full_set[0])
		t_noise = []
		for item in t_input:
			t_noisy_input = []
			for element in item:
				new_element = round(abs(element + random.randint(-2,2) / 10.), 2)
				t_noisy_input.append(new_element)
			t_noise.append(t_noisy_input)
		t_output.append(full_set[1])
		t_matrix = t_noise, t_output
	return t_matrix

test_matrix = gen_test_matrix(training_sets)

count = 0
for test in test_matrix[0]:
    count += 1
    # print "Input: \n", test[0]
    # print "Desired Output: ", test[1]
    print "Test", count, "Actual Output: ", nn.feed_forward(test)



test_set = [0.9, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.9, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.9, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.9, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.9]

test_set2 = [0.1, 0.1, 0.9, 0.1, 0.1,
             0.1, 0.1, 0.9, 0.1, 0.1,
             0.9, 0.9, 0.9, 0.9, 0.9,
             0.1, 0.1, 0.9, 0.1, 0.1,
             0.1, 0.1, 0.9, 0.1, 0.1]

print "SLASH TEST: ", nn.feed_forward(test_set)
print "PLUS TEST: ", nn.feed_forward(test_set2)

out_array = []
for item in test_matrix:
    out_array.append(nn.feed_forward(item[0]))
print out_array


x_array = []
y_array = []
out_array = []

for item in test_matrix:
    x_array.append(item[0])
    y_array.append(item[1])
    out_array.append(nn.feed_forward(item)[0])


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x_array, y_array, out_array, linewidth=0.2, antialiased=True)
plt.savefig("NNplot.png")
plt.show()


plt.plot(error_list)
plt.ylabel('TSSE Error')
plt.savefig('Aggregated TSSE Error.png')
plt.show()


plt.plot(error_rmse)
plt.ylabel('RMSE Error')
plt.show()
plt.savefig('rmse_error.png')


plt.plot(ff_outputs)
plt.ylabel('Test Outputs')
plt.show()
plt.savefig('Test_Outputs.png')
