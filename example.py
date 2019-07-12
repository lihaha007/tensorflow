import tensorflow as tf
import numpy as np

#creat data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

###creat tensorflow structure start###
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

#计算误差
loss = tf.reduce_mean(tf.square(y-y_data))
#梯度下降法传播误差
optimizer =tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
###creat tensorflow structure end###

#创建会话session,激活init
sess = tf.Session()    
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
