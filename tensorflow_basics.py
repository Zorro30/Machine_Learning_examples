import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

#this defined a model.
result = tf.multiply(x1,x2)

print(result)

#create a session to run the command
with tf.Session() as sess:
    output = sess.run(result)
    print(output)