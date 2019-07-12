import tensorflow as tf

matrixl = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrixl,matrix2)    #矩阵乘法

#method1
#sess = tf.Session()
#result = sess.run(product)
#print(result)
#sess.close()

#method2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
