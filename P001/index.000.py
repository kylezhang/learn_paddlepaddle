# coding:utf-8
import paddle.fluid as fluid

a = fluid.layers.fill_constant(shape=[1], value=10, dtype='int64')
# b = fluid.layers.fill_constant(shape=[1], value=10, dtype='int64')

# y = fluid.layers.sum(x=[a])

place = fluid.CPUPlace()
exe = fluid.executor.Executor(place)

exe.run(fluid.default_startup_program())

data = exe.run(program=fluid.default_main_program(), fetch_list=[a])
print(data)
