# 注册并验证生成的 cell kernel

在仓库根目录运行完整示例：

```bash
PYTHONPATH=. python examples/oot_cell_backend.py
PYTHONPATH=. python examples/oot_cell_backend.py --backend triton --device cuda --source /tmp/lif_kernel.py --capture
```

CPU 运行无需 GPU 可选依赖。Triton、TileLang、CuPy 路径使用共享模板生成的真实
kernel。`--source` 在验证前输出可检查的源码；`--compile` 额外检查静态 fullgraph
前向与反向。

示例先取得内置实现，以外部包名称注册，再验证非零初始状态以及 voltage、
charged-voltage 记录 loss。随后使用常规 `LIF(..., backend="tutorial_cell")` API，
在 eval 模式下训练上游 scale。第二个 parity case 检查 AlphaPSC 的状态与记录梯度，
随后 sparse RSNN 将两个已注册模型与现有 connection kernel 组合，并对最终状态添加 loss。
需要保留模型语义、独立调整 kernel 调度的包也可
从这里开始。

## 定位导数错误

Forward 正确不代表训练正确。尝试在外部包中将初始状态梯度错误地归零：

```python
def broken_backward(*args, **kwargs):
    grad_x, grad_states, grad_parameters = implementation.backward(*args, **kwargs)
    grad_states = tuple(None if g is None else torch.zeros_like(g) for g in grad_states)
    return grad_x, grad_states, grad_parameters
```

用不同名称注册这个错误实现，然后验证同一个 `ParityCase`。即使 spike 输出
完全一致，`report.assert_close()` 仍会报告 `grad_v` 和首个失败坐标。
仓库的 `tests/models/kernels/test_oot.py` 执行了这个回归场景。

发布外部包前检查 capability skip，并验证自己声称支持的形状、精度、surrogate、
reset 选项和记录 loss。参见 [callback 与生成契约](../guides/oot_kernels.md)。
