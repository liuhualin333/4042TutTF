import tensorflow as tf
#
# Tutorial 1, Question 1
#


sess = tf.Session()

#define symbolic variables:

x = tf.placeholder(tf.float32)
w = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

synaptic_input = tf.reduce_sum(tf.multiply(w,x)) + b
activation = 1/(1+tf.exp(-0.5*synaptic_input))
output_activation = tf.maximum(0.0,tf.minimum(1.0, synaptic_input))

#define inputs and parameters
inputs = [[1.0, -0.5, 1.0],
          [-1.0, 0.0, -2.0],
          [2.0, 0.5, -1.0]]

w1 = [1.0, -0.5, -1.0]
b1 = 0.0
w2 = [0.0, 2.0, 0.6]
b2 = 0.5
w3 = [-0.5, 0.6]
b3 = 1.0
for inputVector in inputs:
	r1 = (sess.run(activation, {x:inputVector, w: w1, b: b1}))
	r2 = (sess.run(activation, {x:inputVector, w: w2, b: b2}))
	print (sess.run(output_activation, {x:[r1, r2], w: w3, b: b3}))
    
