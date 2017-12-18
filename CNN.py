import tensorflow as tf
#关于卷积，现在的理解就是从全连接神经网络变成了部分连接神经网络
# 通过卷积，大大减少了神经层的参数数量
def weigth_variable(shape):
#定义一个变量w，即权值，并初始化,加入0.1噪声
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#定义卷积，这里卷积步长为1，第一个参数被卷积的大矩阵，第二个参数即
#被卷积的小矩阵，第三个参数为步长，第一个和最后一个必须为一，中间随便，第二个表示
#左右步长，第三个表示上下步长，最后参数表示先将大矩阵全0填充再进行卷积，这里输入矩阵即
#原图片的矩阵，小矩阵即权值矩阵，上下左右步长皆设为一，全0填充
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#定义一个池化层，常用的有最大化池化层和平均池化层，这里两个都定义,池的大小2X2
def avg_pool_2x2(x):
   #第一个参数为需池化的矩阵，第二个为池的维度尺寸（第一个和第二个必须为一，中间两个随便），
    #第三个参数为步长和卷积差不多含义,这里步长采取上下左右都为二，第四个参数是否全0填充
    return tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
def main():
    #从网站上下载数据
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    #创建session会话，tensorflow的图计算全部用session来响应
    sess=tf.InteractiveSession()
    #创建占位符X，Y，之后通过feed方法将值放入占位空间
    x = tf.placeholder("float",shape=[None,784])
    y =tf.placeholder("float",shape=[None,784])
    #开始第一层卷积，这里我们用5X5矩阵进行卷积，卷积以后得出32个通道（特征），被卷积的大矩阵只有一个黑白通道
    W_conv1=weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    x_image=tf.reshape(x,[-1,28,28,1])
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1=avg_pool_2x2(h_conv1)
    W_conv2=weight_variable(5,5,32,64)
    b_conv2=bias_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2=avg_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print ("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
main()
