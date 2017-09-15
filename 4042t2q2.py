import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputList = np.array([[.8,.5,.0], [.9,.7,.3], [1.0,.8,.5], [.0,.2,.3], [.2,.3,.5], [.4,.7,.8]])
yList = np.array([0,0,0,1,1,1])

sess = tf.Session()
x = tf.placeholder(tf.float32)
w = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

synaptic_input = tf.reduce_sum(tf.multiply(w,x)) + b
activation = tf.cond(synaptic_input <= 0, lambda: tf.constant(0.0), lambda: tf.constant(1.0))
error = y -activation

W = np.zeros(3)
for j in range(int(W.shape[0])):
	W[j] = 0.05
B = 0.05

epoch = 100
learningRate = 0.1
for i in range(epoch):
	for idx, element in enumerate(inputList):
		errorVal = (sess.run(error, {x:element, w: W, b: B, y:yList[idx]}))
		for j in range(int(W.shape[0])):
			W[j] += learningRate*errorVal*element[j]
		B = B+learningRate*errorVal
		print ([idx, element, errorVal, W, B])
print((W, B))
errorVal = 0
for idx, element in enumerate(inputList):
	errorVal += (sess.run(error, {x:element, w: W, b: B, y:yList[idx]}))
	print (errorVal)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(inputList[yList==0, 0], inputList[yList==0, 1], inputList[yList==0, 2], 'rx', label = 'class 1')
ax.scatter(inputList[yList==1, 0], inputList[yList==1, 1], inputList[yList==1, 2], 'bx', label = 'class 2')
X = np.arange(0, 1, 0.1)
Y = np.arange(0, 1, 0.1)
X, Y = np.meshgrid(X,Y)
Z = -(W[0]*X + W[1]*Y + B)/W[2]
decision_boundary = ax.plot_surface(X, Y, Z)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.set_title('Decision boundary in Input Space')
plt.show()
