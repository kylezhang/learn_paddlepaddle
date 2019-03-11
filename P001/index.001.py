# coding:utf-8
import numpy as np
import paddle.fluid as fluid

# 定义网络
a = fluid.layers.data(name='a', shape=[1], dtype='float32')
b = fluid.layers.data(name='b', shape=[1], dtype='float32')

result = fluid.layers.elementwise_add(a, b)

# 定义Executor
cpu = fluid.core.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

# 准备数据
data_1 = int(input('Please enter an integer: a='))
data_2 = int(input('Please enter an integer: b='))

x = np.array([data_1])
y = np.array([data_2])

# 执行计算
outs = exe.run(feed={'a': x, 'b': y}, fetch_list=[a, b, result.name])

# 结果
print(x, y, outs)
print("%d + %d = %d" % (data_1, data_2, outs[2][0]))
