import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", [None, 784])
'''
x是一个占位符placeholder，在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，
（这里的None表示此张量的第一个维度可以是任何长度的。） 
'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])

#为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
'''
要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。
TensorFlow在这里实际上所做的是，它会在后台给描述你的计算的那张图里面，增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。
然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本。
'''
init = tf.global_variables_initializer()
#初始化我们创建的变量，此处使用新版本tensorflow的global_variables_initializer()去初始化变量，原版的initialize_all_variables()已经停止使用了
sess = tf.Session()
sess.run(init)
#启动模型，并且初始化变量
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))