import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# create some data
X = np.linspace(0, 1, 1000)
np.random.shuffle(X)    # randomize the data
Y = 3 * X * X * X  +  2 * X * X  + X + np.random.normal(0, 0.05, (1000, ))
# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:700], Y[:700]     # first 700 data points
X_test, Y_test = X[700:], Y[700:]       # last 300 data points

# build the neural network
model = Sequential()
model.add(Dense(input_dim=1, output_dim=10))
model.add(Activation('relu'))
model.add(Dense(output_dim=10))
model.add(Activation('relu'))
model.add(Dense(output_dim=1))
model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(5000):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=300)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

X_test_order = np.sort(X_test)

# # plotting the prediction
Y_pred = model.predict(X_test_order)
plt.scatter(X_test, Y_test)
plt.plot(X_test_order, Y_pred, 'r-')
plt.show()
