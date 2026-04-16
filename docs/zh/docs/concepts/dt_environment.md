# `dt` 环境

btorch 的神经元模型由常微分方程（ODE）定义。为了数值求解这些 ODE，求解器需要一个时间步长 `dt`。与其在每个构造函数和前向调用中传递 `dt`，btorch 使用了一种轻量级计算环境，类似于 BrainPy。

## 设置 `dt`

推荐的模式是使用上下文管理器：

```python
from btorch.models import environ

with environ.context(dt=1.0):
    spikes, states = model(x)
```

这将 `dt` 的作用域限定在前向传递内，避免意外的全局状态泄漏。

## 全局默认值

你也可以设置一个全局默认值（在笔记本或脚本中很有用）：

```python
environ.set(dt=1.0)
```

任何调用 `environ.get("dt")` 的模块都会在没有活跃上下文时回退到此值。

## 忘记设置 `dt` 是常见陷阱

如果未设置 `dt`，神经元前向传递可能会抛出 `KeyError`。错误信息会明确告诉你如何修复：

```
KeyError: 'dt is not found in the context.
You can set it by `with environ.context(dt=value)` locally
or `environ.set(dt=value)` globally.'
```

## 装饰器用法

`environ.context` 也可以作为函数装饰器使用：

```python
@environ.context(dt=1.0)
def forward(model, x):
    return model(x)
```

参阅 [`environ`][btorch.models.environ] 了解完整的环境 API。
