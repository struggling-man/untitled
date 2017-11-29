import tensorflow as tf

# 加一个层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))        # 定义一个矩阵，习惯大写开头,in_size行，out_size列
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)       # 定义列表
    Wx_plus_b = tf.matmul(inputs, Weights) + biases             # matmul矩阵的乘法
    if activation_function is None:                             # 如果是0的话
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs