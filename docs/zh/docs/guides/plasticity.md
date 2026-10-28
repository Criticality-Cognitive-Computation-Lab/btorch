# 在线突触可塑性

通过 `Synapse(plasticity=PairSTDP(...))` 给 `SparseConnection` 配置在线学习。
连接继续持有规范边顺序下的权重，可塑性模块持有迹线状态；学习方程独立于
神经元模型和稀疏执行后端。

```python
import torch
from btorch.models.connection import EdgeWeight, PairSTDP, SparseConnection, Synapse

connection = SparseConnection.from_edges(
    torch.tensor([0, 1]), torch.tensor([1, 0]), 2, 2,
    Synapse(
        weight=EdgeWeight(torch.tensor([0.3, 0.4]), trainable=False),
        delay=torch.tensor([0, 2]),
        plasticity=PairSTDP(
            a_plus=0.01, a_minus=0.012,
            tau_plus=20.0, tau_minus=20.0, w_min=0.0, w_max=1.0,
        ),
    ),
)
```

幅度、时间常数和权重上下界可以是标量或输入边对齐的数组，随权重、受体和
整数延迟一起转换至规范顺序。路由完全相同的重复边必须先显式合并；独立可塑
突触的合并会改变学习语义。

对于边 `e=(i,j,receptor,delay)`，到达事件为 `a_e(t)=z_i(t-delay)`。
每步顺序为：

1. 神经元使用上一步 PSC 更新；当前到达事件使用旧权重传播，保持现有
   `RecurrentNN` 的时序。
2. 旧前、后迹线分别乘 `exp(-dt/tau_plus)` 和 `exp(-dt/tau_minus)`。
3. 用衰减后的旧迹线计算
   `delta=-a_minus*a_e*post_trace_old+a_plus*z_j*pre_trace_old`。
4. 累加完整增量后，仅裁剪一次权重。
5. 将当前到达事件和后神经元脉冲加入各自迹线。

同时发生的脉冲不相互配对，但都能与严格更早的事件配对。新权重用于下一次
传播。不同受体共享后神经元脉冲，但可具有不同参数；逐边迹线保留异质时间常数。

独立使用连接时，调用者管理延迟，先调用 `connection(arrivals)`，再调用
`connection.update_plasticity(arrivals, post_spikes, dt=dt)`。
到达输入布局为 `[..., source*n_delay+delay]`；该方法不会重复推进延迟缓冲区。
在支持的 `RecurrentNN` 中，PSC 管理延迟，递归调度器执行学习。

在线学习在 `torch.no_grad()` 或 `torch.inference_mode()` 下原地修改权重，
与 BPTT、优化器更新分离。启用梯度或梯度检查点时，在推进状态前报错。
eager 基线和原生在线路径支持一个样本和无额外约束的 `EdgeWeight`。共享权重的多样本
归约、结构重连、嵌套 `DelayedPSC`、顶树突/电导递归和 CuPy 可塑性暂不支持；会明确报错，不会静默忽略可塑性。

迹线为持久化 buffer，遵循模块的设备、精度、reset 和 detach 接口。
reset 清空迹线，**不恢复已学习的权重**。完整运行中网络的检查点需要在运行前
以 `functional.init_net_state(..., persistent=True)` 初始化神经元、PSC 和
延迟状态，再保存/加载完整 `state_dict`。分块继续运行时不要 reset。

## 优化的递归执行

神经元/PSC 渲染器与稀疏传播后端分别选择：

```python
from btorch.models import KernelOptions

network.backend = KernelOptions(neuron="tilelang", connection="triton")
# neuron="triton" 也可用；两者渲染相同的 STDP 标量语句。
with torch.no_grad():
    spikes, records = network(inputs)
```

原生在线路径复用现有稀疏传播和 PSC/神经元融合，每步增加一个独立的规范边
STDP 内核，完成迹线衰减、LTD、LTP、合并裁剪和迹线递增。这是性能基线，
**不是 HELION 的单内核时间步**。这里 TileLang 负责神经元、PSC 和可塑性，
稀疏传播仍可使用现有 Triton 路径。静态连接不分配 STDP 迹线，也不执行学习内核。

权重仍由原连接持有。CSR 执行顺序不同时，维护刷新后的执行视图；在线更新
绕开静态打包权重快照。可通过 `update_state_names` 显式记录
`synapse.linear.plasticity.pre_trace`、`synapse.linear.plasticity.post_trace`
及 `synapse.linear.weight.value`，不必在每次运行都保留完整历史。

与其他递归记忆相同，`update_state_names=None` 会逐步记录迹线。
可明确选择所需的神经元/PSC 记录，或用 `update_state_names=()` 关闭历史记录；
最终迹线仍保留在可塑性模块中。

`compute_input_scale` 在每次校准试验前和退出时恢复学习权重，因此校准不会消耗网络的学习状态。

`torch.compile` 通过预分配输出和明确的原地修改声明执行在线推理。
`cudagraph=True` 使用现有 split 分块（默认 `chunk_size=200`），完整块
共享图，余块使用对应长度的图。预热与捕获会恢复迹线和规范权重，不额外学习。
在线 STDP 仍需要 `no_grad`，不能与 BPTT 或梯度检查点同时使用；未配置
STDP 的路径继续保留既有梯度、VJP 和检查点行为。
