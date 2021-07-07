# Issue Info



```
def @main(%data: Tensor[(1, 1024), float32], %fc0_weight: Tensor[(512, 1024), float32], %fc0_bias: Tensor[(512), float32], %fc1_weight: Tensor[(256, 512), float32], %fc1_bias: Tensor[(256), float32], %fc2_weight: Tensor[(128, 256), float32], %fc2_bias: Tensor[(128), float32]) -> Tensor[(1, 128), float32] {
  %0 = nn.dense(%data, %fc0_weight, units=512) /* ty=Tensor[(1, 512), float32] */;
  %1 = nn.bias_add(%0, %fc0_bias, axis=-1) /* ty=Tensor[(1, 512), float32] */;
  %2 = nn.dense(%1, %fc1_weight, units=256) /* ty=Tensor[(1, 256), float32] */;
  %3 = nn.bias_add(%2, %fc1_bias, axis=-1) /* ty=Tensor[(1, 256), float32] */;
  %4 = nn.dense(%3, %fc2_weight, units=128) /* ty=Tensor[(1, 128), float32] */;
  nn.bias_add(%4, %fc2_bias, axis=-1) /* ty=Tensor[(1, 128), float32] */
}

One or more operators have not been tuned. Please tune your model for better performance. Use DEBUG logging level to see more details.
VM Function[0]: main(data)
# reg file size = 19
# instruction count = 22
opcode, fields # inst(text):
 0: 11 0 1   # load_const $1 Const[0]
 1: 16 1 64 2 32 1 1 2   # alloc_storage $2 $1 64 float32 1
 2: 11 1 3   # load_const $3 Const[1]
 3: 5 2 3 2 32 1 2 4 1 512   # alloc_tensor $4 $2 $3 [1, 512] float32
 4: 11 2 5   # load_const $5 Const[2]
 5: 11 3 6   # load_const $6 Const[3]
 6: 4 0 4 1 0 5 6 4   # invoke_packed PackedFunc[0] (in: $0, $5, $6, out: $4)
 7: 11 4 7   # load_const $7 Const[4]
 8: 16 7 64 2 32 1 1 8   # alloc_storage $8 $7 64 float32 1
 9: 11 5 9   # load_const $9 Const[5]
10: 5 8 9 2 32 1 2 10 1 256   # alloc_tensor $10 $8 $9 [1, 256] float32
11: 11 6 11   # load_const $11 Const[6]
12: 11 7 12   # load_const $12 Const[7]
13: 4 1 4 1 4 11 12 10   # invoke_packed PackedFunc[1] (in: $4, $11, $12, out: $10)
14: 11 8 13   # load_const $13 Const[8]
15: 16 13 64 2 32 1 1 14   # alloc_storage $14 $13 64 float32 1
16: 11 9 15   # load_const $15 Const[9]
17: 5 14 15 2 32 1 2 16 1 128   # alloc_tensor $16 $14 $15 [1, 128] float32
18: 11 10 17   # load_const $17 Const[10]
19: 11 11 18   # load_const $18 Const[11]
20: 4 2 4 1 10 17 18 16   # invoke_packed PackedFunc[2] (in: $10, $17, $18, out: $16)
21: 1 16   # ret $16


*** VM style1 runtime time elapsed 0.04385066032409668


def @main(%data: Tensor[(1, 1024), float32], %fc0_weight: Tensor[(512, 1024), float32], %fc0_bias: Tensor[(512), float32], %fc1_weight: Tensor[(256, 512), float32], %fc1_bias: Tensor[(256), float32], %fc2_weight: Tensor[(128, 256), float32], %fc2_bias: Tensor[(128), float32]) -> Tensor[(1, 128), float32] {
  %0 = nn.dense(%data, %fc0_weight, units=512) /* ty=Tensor[(1, 512), float32] */;
  %1 = nn.bias_add(%0, %fc0_bias, axis=-1) /* ty=Tensor[(1, 512), float32] */;
  %2 = nn.dense(%1, %fc1_weight, units=256) /* ty=Tensor[(1, 256), float32] */;
  %3 = nn.bias_add(%2, %fc1_bias, axis=-1) /* ty=Tensor[(1, 256), float32] */;
  %4 = nn.dense(%3, %fc2_weight, units=128) /* ty=Tensor[(1, 128), float32] */;
  nn.bias_add(%4, %fc2_bias, axis=-1) /* ty=Tensor[(1, 128), float32] */
}

VM Function[0]: main(data, fc0_weight, fc0_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
# reg file size = 19
# instruction count = 16
opcode, fields # inst(text):
 0: 11 0 7   # load_const $7 Const[0]
 1: 16 7 64 2 32 1 1 8   # alloc_storage $8 $7 64 float32 1
 2: 11 1 9   # load_const $9 Const[1]
 3: 5 8 9 2 32 1 2 10 1 512   # alloc_tensor $10 $8 $9 [1, 512] float32
 4: 4 0 4 1 0 1 2 10   # invoke_packed PackedFunc[0] (in: $0, $1, $2, out: $10)
 5: 11 2 11   # load_const $11 Const[2]
 6: 16 11 64 2 32 1 1 12   # alloc_storage $12 $11 64 float32 1
 7: 11 3 13   # load_const $13 Const[3]
 8: 5 12 13 2 32 1 2 14 1 256   # alloc_tensor $14 $12 $13 [1, 256] float32
 9: 4 1 4 1 10 3 4 14   # invoke_packed PackedFunc[1] (in: $10, $3, $4, out: $14)
10: 11 4 15   # load_const $15 Const[4]
11: 16 15 64 2 32 1 1 16   # alloc_storage $16 $15 64 float32 1
12: 11 5 17   # load_const $17 Const[5]
13: 5 16 17 2 32 1 2 18 1 128   # alloc_tensor $18 $16 $17 [1, 128] float32
14: 4 2 4 1 14 5 6 18   # invoke_packed PackedFunc[2] (in: $14, $5, $6, out: $18)
15: 1 18   # ret $18


*** VM style2 runtime time elapsed 0.07250618934631348


API 2 is 1.653479988908402 times slower than API 1
```