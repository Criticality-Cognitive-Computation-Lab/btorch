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

`trace_scope="connection"` 为每条规范突触边保存独立迹线。此选项控制学习状态归属，
与合并兼容线性可加 PSC 状态的 `merge_synapse` 相互独立。

时间常数为标量时，`PairSTDP(..., trace_scope="neuron")` 为每个非空
`(source, delay)` 分段保存一条前迹线，为每个神经元保存一条后迹线。
幅度和权重上下界仍可逐边不同。每条事件流只更新一次迹线，孤立后神经元
也保留自己的迹线。恢复检查点时须使用相同布局。此布局支持 eager 和独立原生
更新，独立基线要求 `recurrent="standard", spike_selection="auto"`。
HELION 也支持共享迹线：边先读取旧分段和后神经元迹线，传播屏障后由各域所有者
在同一个时间步内核中更新一次。零延迟融合复用源到分段指针；延迟入边遍历使用
预备的规范边逆映射。标准双内核事件调度也支持零延迟共享迹线：边读取当前迹线，
各域所有者将下一时刻迹线写入独立缓冲区。

稀疏方法和后端均为自动选择时，展开连接维度严格小于 4096 会在内部使用 dense
matmul。学习仍仅更新规范边及声明域的迹线；已有权重写入者同时按规范权重的存储
精度更新预分配 dense 操作数，没有每步矩阵重建或额外复制。此路径复用于不同
原生规则及延迟/受体连接；显式 push/pull、后端或活动发现方式仍保留指定算法。

权重依赖使用
`PowerPairSTDP(PowerPairSTDPOptions(mu_plus=1.0, mu_minus=1.0), ...)`。
它复用 pair 迹线和事件时序。将权重归一化为
`q=clip((weight-w_min)/(w_max-w_min),0,1)`，LTP 乘 `(1-q)**mu_plus`，
LTD 乘 `q**mu_minus`。零指数恢复加性配对，单位指数为乘性依赖。
幅度仍以绝对权重为单位；上下界相等时权重固定，迹线继续演化。
负权重区间也使用数值大小顺序：LTP 朝 `w_max` 移动，会减小抑制强度；
受体类型不隐式改变学习符号。Power 规则提供 eager 参考、独立原生更新和
HELION 融合，均支持 Triton/TileLang；归一化权重律复用加性规则的迹线绑定、
边写入归属和延迟物化。

最近邻配对使用 `NearestPairSTDP(..., pairing="symmetric")`。
每个事件与严格更早的最近一个对侧事件配对，当前事件替换自身迹线；
同一个较早脉冲可以参与多个后续配对。`pairing="restricted"` 还会在事件后
消耗对侧资格，使每个脉冲最多贡献一次 LTP 和一次 LTD。同时事件先使用旧快照，
随后两侧均建立新资格。非零幅度替换迹线；单步内的多个脉冲没有独立时间戳。
配对策略与状态一起保存，由 `state_dict` 恢复。eager 参考和独立
Triton/TileLang 更新共享 pair 权重律和显式迹线策略；
HELION 复用同一策略，以互斥的边写入归属和惰性衰减处理无事件时间步。

独立使用连接时，调用者管理延迟，先调用 `connection(arrivals)`，再调用
`connection.update_plasticity(arrivals, post_spikes, dt=dt)`。
到达输入布局为 `[..., source*n_delay+delay]`；该方法不会重复推进延迟缓冲区。
在支持的 `RecurrentNN` 中，PSC 管理延迟，递归调度器执行学习。

在线学习在 `torch.no_grad()` 或 `torch.inference_mode()` 下原地修改权重，
与 BPTT、优化器更新分离。启用梯度或梯度检查点时，在推进状态前报错。
eager 基线和原生在线路径支持一个样本，以及每条边可独立写入的无额外约束
`EdgeWeight`。支持不重叠的跨步视图；多条边共享同一地址的广播权重会被拒绝。
共享权重的多样本归约、结构重连、嵌套 `DelayedPSC`、顶树突/电导递归和 CuPy 可塑性暂不支持；会明确报错，不会静默忽略可塑性。

迹线为持久化 buffer，遵循模块的设备、精度、reset 和 detach 接口。
reset 清空迹线，**不恢复已学习的权重**。完整运行中网络的检查点需要在运行前
以 `functional.init_net_state(..., persistent=True)` 初始化神经元、PSC 和
延迟状态，再保存/加载完整 `state_dict`。分块继续运行时不要 reset。

## 抑制性稳态学习

`InhibitorySTDP(eta=0.001, tau=20.0, alpha=0.2)` 使用对称配对，并在每次前脉冲
到达时加入恒定的强度减弱项。先衰减旧迹线，计算有符号增量
`polarity * eta * (arrival * (post_old - alpha) + post_spike * pre_old)`，
合并增量后仅裁剪一次，并将当前事件加入迹线。规则参考 [NEST 所述的 Vogels--Sprekeler 模型](https://nest-simulator.readthedocs.io/en/stable/models/vogels_sprekeler_synapse.html)，
但明确使用 btorch 的严格过去迹线约定；同时脉冲不相互配对，到达减弱项仍生效。

默认 `polarity=-1, w_min=-1, w_max=0` 对应负电流权重；正抑制电导可用
`polarity=1, w_min=0, w_max=1`。极性与边界必须一致，即使权重为零也不推断符号。
规则不推断受体类别；不应学习的边设置 `eta=0`。参数可逐边不同；
`trace_scope="neuron"` 复用分段/后神经元迹线，要求标量 `tau`。
重置和序列化沿用 pair 学习的记忆契约。

连续时间不相关放电下的目标率为 `alpha / (2 * tau)`，单位为时间单位的倒数。
`tau=20` 毫秒、`alpha=0.2` 对应连续时间目标 5 Hz；相关活动下不保证网络达到该放电率。
有限 `dt`、独立二值脉冲下，未裁剪增量的平均漂移在
`alpha * expm1(dt / tau) / (2 * dt)` 处变号；不会隐式修正 alpha。
提供 eager 参考实现，以及 Triton/TileLang 独立学习阶段。原生路径复用声明的
规则模板，支持共享迹线、混合存储精度及编译/CUDA Graph 的块间延续。
HELION 和双内核事件调度也复用该模板。延迟连接使用 HELION 扫描；列表调度要求零延迟。

## 电压依赖学习

`VoltageSTDP` 实现不含额外电压延迟的电压门控规则：

```python
from btorch.models.connection import VoltageSTDP, VoltageSTDPOptions

plasticity = VoltageSTDP(VoltageSTDPOptions(
    tau_pre=15.0, tau_plus=7.0, tau_minus=10.0,
    v_rest=-70.0, theta_minus=-70.0, theta_plus=-45.0,
))
# Attach with Synapse(..., plasticity=plasticity).
```

默认 `PostState("v", stage="charged")` 读取当前复位前电压。神经元需开启
对应记录：LIF/GLIF3 使用 `pre_spike_v=True`，Izhikevich 使用
`pre_spike=True`。规则不包含特定神经元方程。也可指定
`voltage=ExternalSignal("post", time_first=True)`，向 `RecurrentNN` 传入
`plasticity_signals={"voltage": voltage_sequence}`；独立连接则显式提供
`signals={"voltage": post_voltage}`。

归一化到达迹线先衰减，再增加 `arrival / tau_pre`。严格更早的迹线值门控连续
LTP，乘以 `dt` 和两个整流电压因子；LTD 仅在到达事件上发生，不乘 `dt`。
两个电压滤波器按当前保持电压做指数更新；LTP 使用右端点求积，合并增量后仅
裁剪一次。每个后神经元均需满足 `theta_plus > theta_minus`。该变体不额外
引入电压滤波延迟或稳态幅度调节。

到达时间常数和幅度为标量或逐边参数；电压时间常数、阈值及复位值为标量或
逐后神经元参数。电压滤波器每个后神经元只更新一次。重置清空到达资格，
将电压滤波器恢复到 `v_rest`，保留学习权重。无新脉冲时仍可能发生 LTP，
不能直接复用 pair 的事件跳过策略。独立 Triton/TileLang 路径每步使用两个
可塑性 kernel：先按后神经元更新电压滤波器，再在旧权重传播后按规范边更新
权重和到达迹线；两阶段复用标量模板渲染器。神经元 charged/state 电压直接
读取当前工作区，外部电压使用既有的常量或时间优先信号接口。除非用户请求
记录，不需要完整电压历史。CUDA Graph 通过既有分块输入刷新外部信号，
原生循环算子也支持 `torch.compile`。使用 `recurrent="helion"` 时，相同阶段
组合为每步一个协作 kernel：神经元和后神经元滤波器先发布结果，经过全网格
屏障后遍历每条规范边，再经过屏障供 PSC 消费累积输入。每条边使用旧权重
传播，再更新连续学习状态。延迟传播沿用独立基线的到达时刻约定。静默边仍
需更新，因此不使用事件跳过或惰性迹线时钟。连续融合要求一个规范边阶段，之前可以有兼容的独立后神经元阶段，
之后可以有独立后神经元或分段阶段；不支持的跨域依赖会明确报错。

## 自定义学习规则

`Synapse.plasticity` 接受继承 `MemoryModule` 的 `PlasticityRule`。
连接持有权重；规则持有注册的状态和固定参数 buffer。规则返回规范边顺序的
权重增量，连接在使用旧权重传播后合并更新，并只裁剪一次。例如：

```python
from btorch.models.connection import ExternalSignal, PlasticityRule

class GatedLTP(PlasticityRule):
    state_domains = {"trace": "edge"}

    def __init__(self):
        super().__init__(
            w_min=0.0, w_max=1.0, signals={"gain": ExternalSignal()},
        )
        self.register_buffer("tau", torch.tensor(20.0))
        self.register_memory("trace", 0.0, (), persistent=True)

    def single_step_forward(self, weight, events, dt):
        old = self.trace.reshape(-1) * torch.exp(-dt / self.tau)
        delta = events.signals["gain"] * events.post_spike * old
        self.trace = (old + events.arrival).reshape_as(self.trace)
        return delta
```

通过 `Synapse(plasticity=GatedLTP(), ...)` 绑定规则。独立连接显式接收实时信号：

```python
with torch.no_grad():
    current = connection(arrivals)  # Old weights.
    connection.update_plasticity(
        arrivals, post_spikes, dt=1.0, signals={"gain": gain},
    )
```

每个状态必须通过 `state_domains` 声明为 `edge`、`segment`（非空来源/延迟分组）、
`post` 或 `scalar`。在 `__init__` 中直接调用
`register_memory(..., sizes=(), persistent=True)`，表示每个域成员一个标量。
连接绑定时自动补上边、分组或神经元的首维；显式 `sizes` 表示域成员内部的尾部维度。
`scalar` 域不增加首维。不需要额外初始化钩子；状态沿用正常的 `init_state`、reset、
设备/精度迁移和检查点行为。
固定参数默认是标量或输入边数组，后者随规范边排序；`parameter_domains` 可声明
后神经元或分组数组。`events.post_index` 和按需提供的 `events.segment_index`
将规范边映射到这些状态域。分组映射复用稀疏表示缓存，仅在规则需要时分配。

自定义张量规则也可用于 eager `RecurrentNN`。声明
`PostState("v", stage="charged")` 后，通过神经元模板的记录元数据读取每步更新的
重置前电压。构造神经元时需要开启对应记录，例如 `LIF(..., pre_spike_v=True)`。
`stage="state"` 则读取步末状态。信号保留神经元布局；逐边计算时用
`events.post_index` 索引。缺失输入在推进状态前报错。外部信号显式传入张量：

```python
with torch.no_grad():
    spikes, records = network(inputs, plasticity_signals={"gain": gain})
```

`ExternalSignal()` 在单次调用内恒定。逐步变化的信号声明为
`ExternalSignal(time_first=True)`，传入 `(T, *domain_shape)` 张量；
例如逐步变化的标量 gain 使用 `(T,)`。`RecurrentNN` 沿用输入的分块和
unroll 调度切分信号，无需第二套时间循环。恒定信号和时间序列可同时传入。
CUDA Graph runner 持有并刷新嵌套输入张量，同形状的新信号张量可复用
捕获。纯张量规则可使用编译，但任意 Python 回调不会自动成为 GPU 模板。
显式选择原生后端时，未支持的规则会在推进状态前报错。

`TripletSTDP(TripletSTDPOptions(...))` 提供四迹线 all-to-all 规则，
包含 pair 项和真正的 triplet 乘积项。它使用严格过去的迹线快照，以及与
`PairSTDP` 相同的延迟到达约定。Triton 和 TileLang 均实现独立优化更新内核
和 HELION 延迟物化融合，复用相同的状态、参数声明和标量公式；
triplet 保留四条独立指数迹线的旧快照。内置电压和调制规则支持其声明的实时信号；
自定义规则需要自己的原生数学模板，否则使用 eager 执行。

## 优化的递归执行

神经元/PSC 渲染器与稀疏传播后端分别选择：

```python
from btorch.models import KernelOptions

network.backend = KernelOptions(neuron="tilelang", connection="tilelang")
# neuron="triton" 也可用；两者渲染相同的 STDP 标量语句。
with torch.no_grad():
    spikes, records = network(inputs)
```

原生在线路径复用现有稀疏传播和 PSC/神经元融合，每块只准备一次规则校验、
张量布局和原生启动绑定。逐边事件规则每步增加一个独立 STDP 内核，完成迹线
衰减、LTD、LTP、合并裁剪和递增。共享事件迹线也只用一个学习内核：边权重读取
旧迹线，分段和后神经元各自写入紧凑的下一份快照。每步交换缓冲区，只有奇数长度
块需要在块末复制回持久迹线。电压规则保留两个有序启动，先更新后神经元滤波状态，
再更新边权重。这是性能基线，
**不是 HELION 的单内核时间步**。稀疏后端也可独立选择：
`connection="triton"` 可将 Triton 传播与 TileLang 神经元、PSC 和可塑性内核组合。
静态连接不分配 STDP 迹线，也不执行学习内核。
状态和系数绑定由所选规则模板声明：pair 使用两条迹线，triplet 使用四条。
合并权重更新和当前事件递增前，先读取全部旧迹线快照。

稀疏方法和后端均自动选择时，展开后的连接矩阵维度严格小于 4096 才使用内部
稠密矩阵乘法。学习仍只处理规范边和声明的迹线域；原有权重写入阶段直接更新
预分配的稠密操作数，使用规范权重的存储精度，无需逐步重建或复制矩阵。
延迟和多受体连接遵循同一规则；显式传播、后端和发现方式保留用户选择。

权重仍由原连接持有。CSR 执行顺序不同时，维护刷新后的执行视图；在线更新
绕开静态打包权重快照。可通过 `update_state_names` 显式记录
`synapse.linear.plasticity.pre_trace`、`synapse.linear.plasticity.post_trace`
及 `synapse.linear.weight.value`，不必在每次运行都保留完整历史。

与其他递归记忆相同，`update_state_names=None` 会逐步记录迹线。
可明确选择所需的神经元/PSC 记录，或用 `update_state_names=()` 关闭历史记录；
最终迹线仍保留在可塑性模块中。

`compute_input_scale` 在每次校准试验前和退出时恢复学习权重，因此校准不会消耗网络的学习状态。

`torch.compile` 通过预分配输出和明确的原地修改声明执行在线推理。
`cudagraph=True` 使用现有分块接口（默认 `chunk_size=200`）。
`chunk_mode="split"` 使用固定长度块，`"chunk"` 使用近似等长块；重复的完整块
共享图，较短的末块直接执行。预热与捕获会恢复迹线和规范权重，不额外学习。
在线 STDP 仍需要 `no_grad`，不能与 BPTT 或梯度检查点同时使用；未配置
STDP 的路径继续保留既有梯度、VJP 和检查点行为。

## 调制资格迹线

`ModulatedSTDP` 将配对产生的资格迹线与学习权重分开保存，通过已有信号接口
显式传入奖励或教学信号：

```python
from btorch.models.connection import ExternalSignal, ModulatedSTDP, ModulatedSTDPOptions

rule = ModulatedSTDP(
    ModulatedSTDPOptions(tau_eligibility=1000.0, eta=0.001),
    modulator=ExternalSignal("scalar", time_first=True),
    trace_scope="neuron",
)
# Attach through Synapse(plasticity=rule), then run:
with torch.no_grad():
    spikes, records = network(inputs, plasticity_signals={"modulator": rewards})
```

`rewards` 的形状为 `[time]`。常量标量使用 `ExternalSignal("scalar")` 和标量张量；
也可声明 edge、post 或 segment 域信号；edge 信号使用规范边顺序。
系数支持标量或输入边数组，共享配对
迹线要求配对时间常数为标量，资格迹线始终逐规范边保存。

每个时间箱内，调制信号保持常量。旧资格迹线 `c` 对权重贡献
`eta * (m-baseline) * c * tau_eligibility * (1-exp(-dt/tau_eligibility))`。
随后资格迹线衰减，并在箱末加入与 `PairSTDP` 一致的严格历史配对脉冲。
同一箱的奖励不会立即作用于新资格；后续奖励即使没有新脉冲也会改变权重。
传播先使用旧权重，合并学习量只裁剪一次。重置清空全部三条迹线而保留已学习权重；
检查点保存全部记忆，恢复时须使用相同迹线布局、调制域和时间轴声明。
系数转换到选定权重 dtype 后必须保持有限，衰减时间必须保持正数。

这是明确规定时间箱顺序的三因素规则，与
[NEST 的多巴胺 STDP](https://nest-simulator.readthedocs.io/en/stable/models/stdp_dopamine_synapse.html)
不同，它直接接收调制量，不滤波体积传递器的脉冲池，也不宣称数值等价。
eager 和独立的 Triton/TileLang 执行复用配对方程。原生执行每步增加一个学习内核，
静默时间步也会更新；传播和神经元/PSC 保留原有调度。共享配对迹线使用紧凑的
旧/新快照，资格迹线在原规范边缓冲区更新。信号域由声明准备，与后端独立。
原生逐点学习保留 half/bfloat16 运算；仅当后端没有低精度除法或数学函数接口时，
临时提升该次运算并立即转回结果，不将整套状态强制转换为 FP32。
混合张量输入遵循 dtype 提升规则；FP32 教学信号不会被降为权重或迹线的 dtype。
两个后端均支持 `recurrent="helion"` 配合 `spike_selection="scan"`：
单个协作内核完成神经元更新、旧权重传播和资格迹线更新；所有边读取完成后再提交共享迹线。
资格迹线在静默时间步也可能改变权重，因此此规则仍遍历所有边，显式的仅事件学习发现方式会报错。
独立基线仍可用 `spike_selection="auto"` 进行稀疏 push 传播。
在线权重更新要求 `torch.no_grad()`；支持 CUDA Graph 重放和编译后的分块续算。

## 实验性的 HELION 执行

配置 `PairSTDP`、`PowerPairSTDP`、`NearestPairSTDP`、`TripletSTDP` 或不启用可塑性时，
均可选择协作式融合时间步：

```python
network.backend = KernelOptions(
    neuron="triton",  # 也可以使用 "tilelang"
    connection_method="push",
    recurrent="helion",
    spike_selection="scan",
)
with torch.no_grad():
    spikes, records = network(inputs)
```

设置 `spike_selection="spike_list"` 可消费已有的神经元直接生成的队列，
代替扫描行。两个后端目前均支持零延迟静态连接和仅由事件驱动的 STDP；
延迟列表执行和连续学习规则会明确报错。两种活动发现方式复用 HELION 的
边遍历；选择列表不会选择标准路径的打包扇出内核。

对于零延迟、仅由事件驱动的 STDP，也可设置 `recurrent="standard"`，
配合任一显式活动发现方式，将 PSC/神经元与传播/可塑性分成两个内核：

```python
network.backend = KernelOptions(
    neuron="triton",
    connection="tilelang",  # 标准递归中可与 neuron 不同
    connection_method="push",
    recurrent="standard",
    spike_selection="spike_list",  # 也可使用 "scan"
)
```

此调度在内部时间步启动两个内核，另有 chunk 边界初始化和最后一次 PSC 更新。
逐边迹线需要块末物化；共享迹线使用紧凑双缓冲，仅奇数长度块结束时复制回持久
状态，没有每步迹线复制或额外迹线内核。目前不支持延迟、`x_syn` 或合并受体状态。
默认的 `spike_selection="auto"` 保留独立 STDP 基线。显式活动发现要求
原生神经元模板和 push 传播；静态 scan 目前要求 `recurrent="helion"`。

`recurrent="helion"` 每个时间步只启动一个 GPU 内核，完成神经元更新、稀疏出边传播和 PSC 更新，
复用模型模板与规范连接权重。传播和 PSC 消费之间使用真正的协作式启动和
设备级屏障；超过协作式启动资源限制时明确报错，不使用不安全的同步。

融合内核统一使用神经元渲染后端；`connection` 必须为 `"auto"` 或与
`neuron` 一致。标准递归路径支持独立选择神经元与稀疏后端。

静态连接的逐边整数延迟使用本次调用内的 FutureCurrent 环形缓冲区。每块开始时从既有
`Delay.history` 重建待到达事件，因此也支持两次调用之间修改权重。
所请求的历史记录保持公开记忆布局，不分配可塑性迹线。CUDA Graph 重放和
编译推理继续使用既有递归记忆接口。此实验路径目前不支持 BPTT、
`x_syn`、`merge_synapse`、连接 bias 或统一 `DelayedPSC` 包装器。
延迟应通过 `Synapse(delay=...)` 配置；原生递归张量路径不支持循环延迟历史。

更少的内核启动次数不保证执行更快。请针对自己的工作负载，与默认的
`recurrent="standard"` 路径进行比较。

配置 pair 或 triplet STDP 时，同一选项选择 D²CSR + PastSpike。按 `(source, delay)`
分段并保留规范边 ID，LTP 复用既有入边索引。第一个屏障发布当前脉冲，随后
处理延迟到达和传播。到达活跃边负责合并 LTD/LTP 更新，入边遍历只处理其余
边，避免权重写冲突；传播读取更新前权重。第二个屏障完成传播和学习，再消费
PSC 输入。两个后端的这些阶段都在同一个时间步内核中执行。

逐边迹线在边参与事件时按自己的时间常数惰性衰减。Triplet 在递增快、慢
迹线前读取全部四条旧快照，沿用 pair 的单写者权重归属。
最近邻配对先使用旧快照计算增量，再替换或消耗资格；
同一边的唯一写者提交两条迹线及其时间戳。
块结束时另用一个内核将最终迹线更新
至当前时刻。初始化、历史记录也有时间步启动计数之外的成本；显式请求每步
迹线历史需要额外遍历边。静态执行不承担这些可塑性开销。

惰性指数衰减与逐步衰减在数学上等价，但浮点舍入可能不同。原子传播也可能
改变累加顺序，因此融合与标准执行不保证产生逐位一致的状态或脉冲栅格。

共享 pair 迹线在所有边读完后，每步按分段和后神经元各更新一次，不需要逐边
迹线时钟或块末物化内核。活动发现使用选定的扫描方式，零延迟时也可使用神经元
直接生成的脉冲列表。静态执行使用 CSR + FutureCurrent，不分配可塑性状态。

## 短时释放

`Synapse.transmission` 通过 `TransmissionRule` 独立描述释放效能，不修改学习权重。
`TsodyksMarkram` 提供短时抑制和易化：

```python
from btorch.models.connection import Synapse, TsodyksMarkram

synapse = Synapse(
    weight=0.5,
    transmission=TsodyksMarkram(U=0.2, tau_rec=800.0, tau_fac=100.0),
)
```

所属 `SparseConnection` 按每条边的实际延迟采集二值到达事件。
规则先让可用资源恢复到一、利用率衰减到零，再在事件到达时提高利用率，
释放 `utilization * resources`，并从资源中扣除该释放量。
`tau_fac=0` 关闭易化。资源和利用率分别初始化为一和零；静默时间步仍执行恢复和衰减。
时间常数与 `environ.context(dt=...)` 使用相同单位。

释放效能只缩放执行时的权重，不修改规范权重。它可以与 `Synapse.plasticity` 同时使用：
传播读取当前学习权重和释放效能，随后的 STDP 更新读取原始到达事件和突触后脉冲，
不会将衰减后的释放量当作 STDP 脉冲输入。重置恢复释放记忆，`state_dict` 保存可续算状态。

当前释放基线在 `torch.no_grad()` 或 `torch.inference_mode()` 下使用 eager 执行。
优化递归内核、BPTT、批量样本、外部稀疏事件输入，以及绑定释放状态后的结构重连均会明确报错。
FP16/BF16 记忆和逐点运算保留原 dtype，混合输入遵循 PyTorch 提升规则。
自定义释放规则通过相同绑定生命周期管理记忆，实现 `step(arrivals, dt)`，
返回每条规范边的释放效能，不修改权重。

### 自定义规则与原生内核的边界

自定义 `PlasticityRule` 通过 `register_memory` 声明持久记忆，通过 `register_buffer`
声明固定系数，使用 `state_domains` 标记记忆属于边、源/延迟分段或突触后神经元。
`single_step_forward(weight, events, dt)` 读取旧状态，计算并返回每条规范边的权重增量，
再提交迹线；不要自行修改权重。所属连接负责在旧权重传播之后应用增量并裁剪一次。
`PostState` 声明实时神经元状态，`ExternalSignal` 声明外部教学信号。

### 树外原生学习模板 { #out-of-tree-native-learning-templates }

直接编写 Triton 或 TileLang 函数，并注册张量绑定。btorch 不解析方程、不进行
符号追踪，也不翻译自定义数学 DSL。规则仍继承 `PlasticityRule`，通过
`register_memory` 声明记忆；具体类设置 `kernel_model = "lab_burst_stdp"`。

```python
from btorch.models.kernels import (
    PlasticityStage, PlasticityTemplate, PlasticityTrace,
    register_plasticity_template,
)
from my_package.kernels import burst_triton, burst_tilelang

register_plasticity_template(
    "lab_burst_stdp",  # BurstSTDP.kernel_model
    PlasticityTemplate(
        states=("pre_trace", "post_trace"),
        parameters=("a_plus", "a_minus", "tau", "w_min", "w_max"),
        stages=(PlasticityStage(
            domain="edge",
            reads=("weight", "pre_trace", "post_trace", "a_plus", "a_minus",
                   "tau", "w_min", "w_max", "arrival", "post_event", "dt"),
            outputs=("weight", "pre_trace", "post_trace"),
            triton=burst_triton,
            tilelang=burst_tilelang,
        ),),
        traces=(PlasticityTrace("pre_trace", "tau", "arrival"),
                PlasticityTrace("post_trace", "tau", "post_event")),
        event_driven=True,
    ),
)
```

`burst_triton` 是普通 `@triton.jit` 辅助函数；`burst_tilelang` 是在构建内核时
返回标量表达式的 TileLang 函数。输入顺序由 `reads` 指定，返回 tuple 的顺序由
`outputs` 指定。函数运行在既有内核内部，不是每步执行的 Python 回调。可以只提供
一个后端；明确选择未提供的后端时，会在推进模拟状态之前报错。原生函数负责数学、
裁剪与精度，保留输入张量的 dtype，仅提升确实需要提升的运算。

独立学习阶段和 HELION 使用 `neuron` 后端。在标准递归中显式选择
`spike_selection` 时，传播和学习共同使用 `connection` 后端；
`connection="auto"` 则继承 `neuron`。应为实际执行学习的后端提供原生函数，
即使产生脉冲的神经元使用另一个后端。

使用 `event_driven=True` 时，在上述输入之后增加两个编译期布尔参数
`include_ltp`、`include_ltd`。它们仅控制权重项；迹线仍根据两个原始当前事件更新。
同时发生的脉冲读取已衰减的旧迹线，权重增量合并后只裁剪一次；连接先以旧权重
传播，再提交学习更新。btorch 利用此契约维持出边/入边遍历中每条边的唯一写者。

`PlasticityTrace` 声明精确、独立的指数衰减与加性事件累积，允许惰性推进静默边、
在数学允许时共享迹线域，并在分块边界物化状态。它不实现或推断用户方程。
只有静默步权重不变、原生迹线更新满足这些契约时，才能声明 `event_driven=True`。
目前原生事件钩子要求单个 edge 阶段、全部记忆都有上述迹线契约，且不使用实时信号。
其他规则省略 `event_driven`，使用普通原生阶段，并显式声明状态/参数域和信号。
既有连续 HELION 调度支持兼容的跨域依赖；不支持的调度会明确报错。

相同函数可用于普通更新、稀疏事件传播和支持的 HELION 融合。记忆、重置、序列化、
规范权重身份、记录和分块续跑仍由 btorch 管理。在线 STDP 要求 `no_grad` 或
`inference_mode`，不注册 BPTT backward 回调。

运行 `python examples/oot_stdp.py` 查看完整的独立 Torch 规则及原生函数。
使用 `--backend triton --device cuda --capture`，或
`--backend tilelang --device cuda --recurrent helion --capture`，比较 eager Torch 的
脉冲、神经元/PSC 状态、迹线及权重。`--compile` 检查原生递归算子，`--dtype` 选择精度。

在编译/捕获之前注册。重复名称报错；`replace=True` 只替换外部注册并递增版本，
不能替换内置规则。RecurrentNN 自动使旧捕获失效；替换后应重新创建编译调用和独立
CUDA Graph。旧版本在线算子会在状态修改前拒绝执行。子类必须显式声明自己的
`kernel_model`。可使用 `plasticity_template_names` 与
`plasticity_template_revision` 查询注册信息。


## 独立 Izhikevich 示例

运行 `python -m examples.rsnn_izhikevich_stdp`，使用 60K 神经元回路、
兴奋性成对学习、固定抑制权重和分块 CUDA Graph 重放。
分别通过 `--recurrent standard|helion` 和 `--spike-selection scan|spike_list`
选择调度与事件发现方式。示例记录神经元/受体历史，不分配逐步学习历史。
计时范围、Torch 参考及静态对照见[示例指南](../examples.md)。

## 复现冷启动与热启动对比

在仓库根目录运行 `python -m benchmarks.rsnn_izhikevich.run --modes inference stdp`。
脚本复用 Izhikevich 示例，测试 4096 和 60,000 个神经元，包含 Torch native
有、无 CUDA Graph，以及 Triton/TileLang 的 push 和 HELION。每个组合启动独立
进程，允许使用磁盘编译缓存。冷时间包含有限长度分块的 JIT、预热、捕获和实际运行；
建网、重置、校验及写盘另计。结果保留各项测量、匹配组合的几何平均加速比、减速和失败。
训练、计时范围及不支持的组合见
[benchmark 说明](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/stdp/benchmarks/rsnn_izhikevich/README.md)。
