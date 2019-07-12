import tensorflow as tf

#定义语法 state = tf.Variable()
state = tf.Variable (0,name='counter')   
#print(state.name)

#定义常量one
one = tf.constant(1)

#定义加法
new_value = tf.add(state,one)
#将state更新为new_value
update = tf.assign(state,new_value)

#若设定了变量，初始化变量是非常重要的，即如果定义Variable,就一定要initialize
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state)) #yi定要把sess的指针指向state再进行print才能得到结果
