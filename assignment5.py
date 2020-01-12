import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(a):
    return 1 / (1 + np.exp(-0.005*a))


def sigmoid_derivative(a):
    return 0.005 * a*(1 - a)


def run_on_test_set(test_inputs, test_labels, weights):
    test_outputs = np.array(np.dot(test_inputs, weights), dtype=np.float32)
    test_outputs = sigmoid(test_outputs)
    tp = 0
    for i in range(test_outputs.shape[0]):
        for j in range(test_outputs.shape[1]):
            if test_outputs[i][j] > 0.5:
                test_outputs[i][j] = 1
            else:
                test_outputs[i][j] = 0

    test_predictions = test_outputs.astype(int)

    for predicted_val, label in zip(test_predictions, test_labels):
        if predicted_val == label:
            tp += 1
    accuracy = tp / test_predictions.shape[0]
    return accuracy


data = pd.read_csv('breast-cancer-wisconsin.csv')

row_count = data.shape[0]   # split data
split_point = int(row_count*0.2)
test_set, train_set = data[:split_point], data[split_point:]
train_set = train_set.replace('?', 1)   # complete missing data
test_set = test_set.replace('?', 1)

train_set = train_set.drop('Code_number', axis=1)
test_set = test_set.drop('Code_number', axis=1)

f = plt.figure(figsize=(10, 8))     # draw heat map
plt.matshow(train_set.corr(), fignum=f.number)
plt.xticks(range(train_set.shape[1]-1), train_set.columns, fontsize=7, rotation=45)
plt.yticks(range(train_set.shape[1]-1), train_set.columns, fontsize=7)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
labels = train_set.corr().values
for y in range(train_set.corr().shape[0]):
    for x in range(train_set.corr().shape[1]):
        plt.text(x, y, '{0:.2f}'.format(labels[y, x]), ha='center', va='center', color='white')
plt.show()

train_temp, test_temp = np.array(train_set), np.array(test_set)  # find test,train;input,output
train_inputs, test_inputs = train_temp[:, :-1], test_temp[:, :-1]
train_outputs, test_labels = train_temp[:, 9:], test_temp[:, 9:]
train_inputs, test_inputs = train_inputs.astype(float), test_inputs.astype(float)
iteration_count = 2500
np.random.seed(1)
weights = 2 * np.random.random((9, 1)) - 1
accuracy_array, loss_array = [], []

for iteration in range(iteration_count):
    outputs = train_inputs.dot(weights)
    outputs = sigmoid(np.array(outputs, dtype=np.float32))
    loss = train_outputs - outputs
    tuning = loss*sigmoid_derivative(outputs)
    weights = np.add(weights, np.dot(train_inputs.T, tuning))
    accuracy_array.append(run_on_test_set(test_inputs, test_labels, weights))
    loss_array.append(np.mean(loss))

iteration_list = [k for k in range(2500)]

fig = plt.figure()
x1 = fig.add_subplot(2, 1, 1)
plt.plot(iteration_list, accuracy_array)
plt.ylabel("Accuracy")
x2 = fig.add_subplot(2, 1, 2)
plt.plot(iteration_list, loss_array)
plt.xlabel("#Epochs")
plt.ylabel("Loss")
plt.show()
