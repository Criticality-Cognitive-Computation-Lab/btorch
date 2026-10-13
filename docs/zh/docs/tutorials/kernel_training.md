# 使用 cell 后端训练 sparse RSNN

完整示例包含可训练 encoder、独立的 sparse recurrent connection、LIF/AlphaPSC
以及可训练 readout：

```bash
PYTHONPATH=. python examples/kernel_backend_training.py
PYTHONPATH=. python examples/kernel_backend_training.py --backend triton --device cuda
```

可将 cell 名称改为 `tilelang` 或 `cupy`。Connection 仍由已有 sparse runtime
执行，后端独立选择。

示例针对 `[T,B,N]` 输入初始化 FP32 状态，在 inference_mode 下 inference，
reset 后连续训练两段，再统一 backward。Voltage 和 PSC 记录参与附加 loss。
`eval()` 保留 encoder、neuron、readout 梯度；chunk 边界不会隐式 detach，
backward 后显式调用 `functional.detach_net` 结束轨迹。

Checkpoint 可继续使用现有 `chunk_size`、`unroll`、`grad_checkpoint` 选项；
chunk size 应与 unroll 兼容。切换 receptor merge 前验证连续状态与各项 loss，
切换 merge 模式需要 reset。

后端选项参见 [KernelOptions 指南](../guides/kernel_backends.md)；外部实现的
验证参见 [OOT parity 教程](oot_kernels.md)。
