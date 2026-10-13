# 网络专用 SONATA IO

每个网络提供自己的 SONATA adapter。btorch 不遍历任意 Module 来推断网络定义；用户的 agent 负责遍历、模型构建、节点/边映射及单位转换。`btorch.io.sonata` 共享 helper 只处理单属性组 HDF5 population 和原生边索引。

统一调用接口为 `SonataNetworkIO`，无需继承或注册：

```python
from examples.sonata_izhikevich import IzhikevichSonataIO, make_example
from btorch.models import environ, functional

adapter = IzhikevichSonataIO()
config = adapter.export_network(make_example(), directory, dt=0.25)
loaded = adapter.import_network(config, device="cpu", backend="torch")
with environ.context(dt=loaded.dt):
    functional.init_net_state(loaded.network, batch_size=1)
    spikes, records = loaded.network(x_seq)
```

导出返回 circuit config 路径，导入返回 `SonataImport(network, dt)`，不修改 environ 或初始化运行状态。

```bash
pip install 'btorch[io]' libsonata
python examples/sonata_izhikevich.py --directory ./sonata-izh
```

示例支持单个 Izhikevich population、AlphaPSC、普通 sparse 逐边权重及标量/逐神经元参数；通过版本化 `btorch` config 扩展保留 reset、surrogate、可训练性、记录请求和单位。节点 ID 对应 neuron 位置；source 是 pre，target 是 post。零 edge delay 表示没有额外传播延迟，RecurrentNN 仍使用上一时刻 PSC。其他模型、receptor、delay、参数共享及自定义机制由对应 adapter 实现，不支持的配置报错。

文件遵循 [SONATA 规范](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md)。索引用 [libSONATA](https://libsonata.readthedocs.io/en/latest/api.html) 生成，并用 hard link 兼容 pySONATA 的索引名称。测试通过两个 reader 检查拓扑、属性、入边/出边查询，并比较原网络与恢复网络的活动和梯度。可选安装官方 [pySONATA](https://github.com/AllenInstitute/sonata/tree/master/src/pysonata)：

```bash
pip install 'git+https://github.com/AllenInstitute/sonata.git#subdirectory=src/pysonata'
PYTHONPATH=. python -m pytest tests/io/test_sonata.py
```

pySONATA 的 node dynamics accessor 尚未实现，该部分由 libSONATA 验证。能读取文件不代表其他 simulator 能执行自定义 btorch 模型；运行中的膜电位、PSC、delay 和 optimizer continuation 使用独立的 btorch checkpoint。

用户指定 SONATA，或需要网络定义序列化但未指定 IO 格式/backend 时，kernel-gen harness 调用 [sonata-io skill](../../../../skills/sonata-io/SKILL.md)。沿用统一接口和共享 helper，具体网络实现交给用户的 agent。只 upstream 共享 helper、接口和可复用检查，生成的网络专用实现留在用户项目/OOT 包中。
