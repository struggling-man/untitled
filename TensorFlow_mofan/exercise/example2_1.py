import tensorflow as tf

# create two matrixes

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)        # matrix multiply, np.dot(m1, m2)


# method 1
sess = tf.Session()
result = sess.run(product)      # run以下才执行一次
print(result)
sess.close()
# [[12]]

# method 2
with tf.Session() as sess:          # with 语句，把这个赋值给它，此处不用关闭，相当于for，用完就关了
    result2 = sess.run(product)
    print(result2)
# [[12]]


# 定义变量语法，定义了某字符串是变量，它才是变量
state = tf.Variable(0, name='counter')      # 变量是0，name is counter
# print(state.name)

# 定义常量one
one = tf.constant(1)

# 定义加法步骤
new_value = tf.add(state, one)

# 将state更新成new_value
update = tf.assign(state, new_value)

# init = tf.initialize_all_variables()        # 初始化所有变量，must have if define variable
init = tf.global_variables_initializer()       #初始化变量方式改为这个了

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))