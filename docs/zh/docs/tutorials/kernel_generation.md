# 生成 kernel 并贡献到 btorch

kernel-gen harness 先验证实际 Torch 模型，再检查 kernel 正确性、测量性能并验证完整 workload。入口可以是模型描述、已有 Python 模型或 OOT kernel；支持自定义 neuron、融合网络、sparse format、量化和不同 backward 算法。

## 配置外部工具

使用已安装的 [Humanize](https://github.com/PolyArch/humanize)、[KernelWiki](https://github.com/DongyunZou/KernelWiki) 和 [ncu-report-skill](https://github.com/DongyunZou/ncu-report-skill)。它们是外部依赖，不是仓库 submodules；优先复用现有安装，否则按 `skills/kernel-gen/README.md` 和各上游说明安装。通过 agent 支持的 discovery 加载其 skills 与 `skills/kernel-gen`。

在任务工作目录编写 `docs/draft.md`，运行 `/humanize:gen-plan` 和 `/humanize:start-rlcr-loop`。Phase 1 产生正确实现；Phase 2 根据测量探索优化；Phase 3 验证完整 workload，加入有收益依据的特化。硬件与预算由任务确定。

## 先验证 Torch 模型

使用 `skills/kernel-gen/assets/templates/glif3_rsnn.py`，实例化实际 GLIF3、AlphaPSC 和 SparseConnection，由 RecurrentNN 执行序列。同一 factory 用于单神经元、上下游两个神经元和完整网络；不另写时间循环 simulator。`stateful_module.py` 展示自定义模型的 register_memory 用法，不重新实现内置 neuron。

检查无输入、脉冲、固定输入和停止输入，再检查连接/delay 与网络响应。三个层级必须有活动图，覆盖全部相关 adaptation/receptor、输入、spikes、膜电位和 PSC。参考 `tests/models/neurons/test_neurons.py` 的 GLIF3 实践。

新生成或修改的模型由用户检查实际结果和图后确认，agent 不能代替确认。内置模型使用现有 btorch Python Module 作为 golden，不另写方程。

参考 [NESTML](https://nestml.readthedocs.io/en/latest/) 的状态、单位和事件表达，以及 [STDP windows](https://nestml.readthedocs.io/en/latest/tutorials/stdp_windows/stdp_windows.html)、[随机模型](https://nestml.readthedocs.io/en/latest/tutorials/gl_model/gl_model_tutorial.html)的验证方法，Torch 模型仍是实现入口。

```python
# MemoryModule template: registered state and a single-step rule.
self.register_memory("q", 0.0, n_neuron)

def single_step_forward(self, x):
    self.q = self.q + environ.get("dt") * self.gain * x
    return self.q
```


MemoryModule 模板的可运行用法：先初始化状态，再调用它自己的连续序列入口。

```python
import runpy
import torch
from btorch.models import environ, functional

CumulativeDrive = runpy.run_path(
    "skills/kernel-gen/assets/templates/stateful_module.py"
)["CumulativeDrive"]
model = CumulativeDrive(3, device="cpu", dtype=torch.float32)
x_seq = torch.ones(10, 2, 3, dtype=torch.float32)
functional.init_net_state(model, batch_size=2, device="cpu", dtype=x_seq.dtype)
with environ.context(dt=1.0):
    output_seq = model.multi_step_forward(x_seq)
```

基础 neuron/PSC 的 `multi_step_forward` 承载连续序列语义，也可以调用注册的 kernel/op。RecurrentNN 负责网络时序组合，当前 step scheduler 主要调用单步方法，将来也可以消费基础多步 op。不要用 `make_rnn` 替代基础模型的序列入口。需要状态 history 时，显式 adapter 必须暴露该入口实际返回的记录。尤其要检查 PSC 各路径的初态/终态契约：现有 Torch 卷积路径无状态，不能与 stateful continuation 偷换。

## 区分三条执行路径

| 路径 | 职责 |
|---|---|
| Inference / inference | 计算输出并更新状态，只保留用户记录，不保存 backward 专用 tape |
| Training forward | 计算输出/最终状态，保留或重算 backward 所需信息 |
| Backward | 使用对应 forward 的上下文，计算输入、初始状态和参数的 VJP |

`.train()`/`.eval()` 不决定路径。整个调用按实际 autograd 需求选择；`no_grad()`/`inference_mode()` 使用 inference。参数冻结但上游 encoder 需要梯度时仍需 training forward。Inference 仍维护模型状态。

使用 Torch 原生 custom operator、fake/meta 和 autograd 注册。Backward 可以使用另一种算法。`native_operator.py` 提供可运行的独立入口与 `KERNEL_BINDINGS` 模板，其 affine 计算仅演示 API，不作为神经模型 golden。

以下两个片段在仓库根目录的同一 Python 进程中运行，只验证 affine API 模板，不构成 neuron 或 RSNN kernel 的验证。

```python
from btorch.models.kernels.native import load_kernel_package

import torch

bindings = load_kernel_package("skills/kernel-gen/assets/templates/native_operator.py")
x = torch.randn(5, 2, 3, requires_grad=True)
gain = torch.ones(3, requires_grad=True)
result = bindings["affine"](x, gain)
```

实际参数和初始状态显式作为 tensor 输入，Module adapter 负责状态写回。

## 验证 parity

```python
from btorch.models.kernels.native import (
    ModelParityCase, ModelParityOptions, check_model_parity,
)

report = check_model_parity(
    [ModelParityCase(
        "broadcast_gain", lambda x, gain: x * gain,
        bindings["affine"], (x, gain),
    )],
    options=ModelParityOptions(atol=1e-5, rtol=1e-5),
)
report.assert_close()
```

Golden 和 candidate 从独立复制的输入、参数和初始状态执行。返回的最终状态和记录也参与 VJP。案例包括 delay wrap、chunk、共享参数以及多个 forward 后 backward 等相关边界。

只验证 inference kernel 时，使用 `ModelParityOptions(modes=("inference", "no_grad"), check_gradients=False)`。这是选择测试上下文，不是运行模式开关；输入全部冻结时，开启 grad 的运行也不能证明执行了 training kernel。

`opcheck` 检查注册，适用时用 `gradcheck` 检查数值导数。Surrogate/STE 不是硬阈值的有限差分导数；EventProp 等不同梯度定义应提供 `gradient_oracle`。量化和随机模型分别声明 reference 与精度/统计契约，不把缺少验证算作通过。

## 执行与反馈

Suite callable 返回 `checks`、`figures`、`source_files`、`required_checks` 和执行/梯度 `metadata`；检查标明 `single`、`pair` 或 `network`、状态与具体结果。

`plot_response` 对齐 input、raster 和状态面板，分开 adaptation/receptor 分量，并保持 neuron 颜色一致。通过 `ResponsePlotOptions(units=..., time_unit="ms", batch_index=0)` 显式指定单位和 trial。示例对图和 JSON evidence 使用同一份单位；未知 state layout 保留所有索引值，不猜测 neuron 身份，也不为美观裁剪或归一化数据。

```bash
python skills/kernel-gen/scripts/check_model.py verify --suite examples/kernel_generation.py:run_suite --output run/report.json
# 仅在用户检查并确认新模型后执行：
python skills/kernel-gen/scripts/check_model.py confirm --report run/report.json --reviewer YOUR_NAME --output run/confirmation.json
python skills/kernel-gen/scripts/check_model.py gate --report run/report.json --confirmation run/confirmation.json
```

用规划和 review 工具记录候选与优化过程；本补充仅提供模型验证和 upstream 工具。反馈应给出首次分歧时间、状态、索引、数值和图/trace 路径。分别测量 inference、training forward、backward、显存及编译/capture 成本。OOT 输出路径不受仓库图目录规则限制。

Evidence gate 只检查声明的结果和文件完整性，不能证明科学正确性、认证 reviewer 身份或证明图来自本次运行。必须查看 suite 断言、实际 trace 和活动图。内置 GLIF3 示例仅检查有限值、静默期与 spike 是否存在，不验证 delay、adaptation dynamics 或 kernel parity。

## Upstream 与默认发布

在目标 btorch 的独立 worktree 中接入同一份实现，保留无关修改。沿用模型、kernel、公共导出和 tests 的现有组织。仅 upstream 时使用 `fig_path()`/`save_fig()`；测试传入 `file=__file__`，避免公共 helper 改变图归档位置。

必须使用目标 btorch 的 `.pre-commit-config.yaml`：安装 hooks，执行 `pre-commit run --files ...`，应用修改后重复至通过；格式化后重新执行相关测试、Ruff 和中英文文档检查。测试覆盖真实用法和边界，附解释性注释。

证据与格式检查通过后，用普通 Git 和 GitHub/GitLab CLI 提交选定文件、push 分支，默认创建或更新 Draft PR/MR；本补充不实现发布自动化。包含三层活动图、parity、性能/profile 摘要、复现方式和来源/许可证。不自动合并；缺少目标或认证时保留本地修改并报告具体阻塞。

示例（替换为实际路径和目标分支）：

```bash
git worktree add -b contribution/glif3-kernel ../btorch-contribution origin/main
cd ../btorch-contribution
pre-commit install --install-hooks
pre-commit run --files btorch/models/neurons/my_model.py tests/models/neurons/test_my_model.py
pytest tests/models/neurons/test_my_model.py
git add btorch/models/neurons/my_model.py tests/models/neurons/test_my_model.py
git commit -m "Add validated neuron model"
git push -u origin contribution/glif3-kernel
gh pr create --draft --title "Add validated neuron model" --body-file pr.md
# GitLab 替代，使用安装版本支持的 description 参数：
glab mr create --draft --title "Add validated neuron model" --description "$(cat pr.md)"
```

脚本仅检查科学证据，规划/review 直接使用 Humanize。只验证声明的执行路径，inference-only kernel 不要求未请求的 backward。模型确认关联模型、案例、图和原始数据版本；仅时间戳或性能指标变化不会使确认失效。
