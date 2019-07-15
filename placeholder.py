import tensorflow as tf 

#placeholder是Tensotflow中的占位符，暂时储存变量
#需要定义placeholder的type,一般为float32
imput1 = tf.placeholder(tf.float32)
imput2 = tf.placeholder(tf.float32)

output = tf.multiply(imput1,imput2)

#传输数据形式:sess.run(**,feed_dict={imput:**})
with tf.Session() as sess:
    print(sess.run(output,feed_dict={imput1:[7.],imput2:[2.]}))
