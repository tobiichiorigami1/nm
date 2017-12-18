import tensorflow as tf
input1 = tf.constant([1.0,2.0,3.0],name="input1")
input2 = tf.Variable(tf.random_uniform([3]),name="input2")
output = tf.add_n([input1,input2],name="add")
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
sess.run(input2)
writer = tf.summary.FileWriter("/path/to/log",tf.get_default_graph())
#sess=tf.Session()
#sess.run(output)
writer.close()

