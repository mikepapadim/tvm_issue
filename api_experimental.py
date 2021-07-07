import tvm
import tvm.relay as relay
import datetime
import time
from tvm.relay.testing import layers
from tvm.relay.testing import create_workload
import numpy as np
import tvm.runtime.vm as vm_rt

dtype = "float32"
target = 'llvm'
batch_size = 1
level=2
warm_iterations=1000
measurements=100

def evaluate_vm_runtime_style1(batch_size):
    mod, params = build_net(batch_size)
    with tvm.transform.PassContext(opt_level=level):
        print(mod)
        exe = relay.vm.compile(mod, target=target, params=params)
        print(exe.bytecode)
        vm = vm_rt.VirtualMachine(exe, tvm.cpu(0))
        input_shape = (batch_size, 1024)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        input_list = [data_tvm]
        for i in range(warm_iterations):    # warm up
            vm.run(input_list)
        vm.run(input_list)
        start_time = time.time()
        for i in range(measurements):
            vm.run(input_list)
        end_time = time.time()
        tvm_time = end_time - start_time
        print("*** VM style1 runtime time elapsed", tvm_time)
        print("\n")
    return tvm_time

def evaluate_vm_runtime_style2(batch_size):
    mod, params = build_net(batch_size)
    with tvm.transform.PassContext(opt_level=level):
        print(mod)        


        executor = relay.build_module.create_executor("vm", mod, tvm.cpu(0), target, params)

        input_shape = (batch_size, 1024)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        input_list = [data_tvm]
        for i in range(warm_iterations):    # warm up
            executor.evaluate()(data_tvm)
        start_time = time.time()
        for i in range(measurements):
            executor.evaluate()(data_tvm)
        end_time = time.time()
        tvm_time = end_time - start_time
        print("*** VM style2 runtime time elapsed", tvm_time)
        tvm_time=0
        print("\n")
    return tvm_time

def build_net(batch_size):
    input_shape = (batch_size, 1024)
    data = relay.var("data", shape=input_shape, dtype=dtype)
    dense0 = layers.dense_add_bias(data=data, units=512, name='fc0')
    dense1 = layers.dense_add_bias(data=dense0, units=256, name='fc1')
    dense2 = layers.dense_add_bias(data=dense1, units=128, name='fc2')
    func = relay.Function(relay.analysis.free_vars(dense2), dense2)
    mod, params = create_workload(func)
    return mod, params

vm_1_time = evaluate_vm_runtime_style1(1)

vm_2_time = evaluate_vm_runtime_style2(1)
print("API 2 is {} times slower than API 1".format(vm_2_time/vm_1_time))
