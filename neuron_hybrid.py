from typing import Callable, Union
import torch
from spikingjelly.activation_based import surrogate
from . import neuron_kernel_float, neuron_kernel_tensor
from .base import MemoryModule
from .jit_utils import jit_script


class BaseNode(MemoryModule):
    def __init__(
        self,
        n_neuron: int,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        assert isinstance(detach_reset, bool)
        super().__init__()

        self.n_neuron = n_neuron

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

    def extra_repr(self):
        return f"detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}"

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y, _ = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq), self.v_seq


class AdaptBaseNode(BaseNode):
    def __init__(
        self,
        n_neuron: int,
        v_min: Union[float, torch.Tensor],
        v_peak: Union[float, torch.Tensor],
        v_r: Union[float, torch.Tensor],
        u_rest: Union[float, torch.Tensor],
        a: Union[float, torch.Tensor],
        b: Union[float, torch.Tensor],
        d: Union[float, torch.Tensor],
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        super().__init__(
            n_neuron, surrogate_function, detach_reset, step_mode, backend, store_v_seq
        )
        # 1. 定义记忆变量的“重置行为”（你已有的代码）
        # 这告诉 reset() 函数：'v' 和 'u' 是需要重置的记忆状态
        if v_min is None:
            self.register_memory("v", 0.0, n_neuron)
        else:
            self.register_memory("v", v_min, n_neuron)

        self.register_memory("u", u_rest, n_neuron)

        if v_min is None:
            # 情况一：软重置 (v_min 是 None)
            # 此时我们需要根据一个 *数字* (0.0) 来创建一个全新的张量。
            # 在这种情况下，使用 torch.full 是完全正确的，因为 0.0 是一个标量。
            self.register_buffer("v", torch.full((n_neuron,), 0.0))
        else:
            # 情况二：硬重置 (v_min 是一个张量)
            # 我们不需要再用 torch.full 创建新张量，因为 v_min 本身就是我们想要的初始状态。
            # 直接克隆 v_min 张量并注册为缓冲区即可。
            self.register_buffer("v", v_min.clone().detach())
        
        # 为适应性变量 u 创建并注册缓冲区。
        # u_rest 已经是一个张量，所以我们同样直接克隆它。
        self.register_buffer("u", u_rest.clone().detach())
        
        self.register_buffer("v_min", v_min.clone().detach(), persistent=False)
        self.register_buffer("v_peak", v_peak.clone().detach(), persistent=False)
        self.register_buffer("v_r", v_r.clone().detach(), persistent=False)
        self.register_buffer("u_rest", u_rest.clone().detach(), persistent=False)
        self.register_buffer("a", a.clone().detach(), persistent=False)
        self.register_buffer("b", b.clone().detach(), persistent=False)
        self.register_buffer("d", d.clone().detach(), persistent=False)
        
        '''
        self.register_buffer("v_min", torch.tensor(v_min), persistent=False)
        self.register_buffer("v_peak", torch.tensor(v_peak), persistent=False)
        self.register_buffer("u_rest", torch.tensor(u_rest), persistent=False)
        self.register_buffer("v_r", torch.tensor(v_r), persistent=False)
        self.register_buffer("a", torch.tensor(a), persistent=False)
        self.register_buffer("b", torch.tensor(b), persistent=False)
        self.register_buffer("d", torch.tensor(d), persistent=False)
        '''

    @staticmethod
    @jit_script
    def jit_neuronal_adaptation(u: torch.Tensor, a, b, v_r, v: torch.Tensor):
        return u + a * (b * (v - v_r) - u)

    def neuronal_adaptation(self):
        """
        Spike-triggered update of adaptation current.
        """
        self.u = self.jit_neuronal_adaptation(self.u, self.a, self.b, self.v_r, self.v)

    @staticmethod
    @jit_script
    # The jit_hard_reset expects parameters the type of v_min and d is torch.Tensor.
    def jit_hard_reset(
        v: torch.Tensor,
        u: torch.Tensor,
        spike_d: torch.Tensor,
        v_min,
        d,
        spike: torch.Tensor,
    ):
        v = (1.0 - spike_d) * v + spike * v_min
        u = u + d * spike
        return v, u

    @staticmethod
    @jit_script
    # The jit_soft_reset expects parameters the type of v_peak and d is torch.Tensor.
    def jit_soft_reset(
        v: torch.Tensor,
        u: torch.Tensor,
        spike_d: torch.Tensor,
        v_peak,
        d,
        spike: torch.Tensor,
    ):
        v = v - spike_d * v_peak
        u = u + d * spike
        return v, u

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_min is None:
            # soft reset
            self.v, self.u = self.jit_soft_reset(
                self.v, self.u, spike_d, self.v_peak, self.d, spike
            )

        else:
            # hard reset
            self.v, self.u = self.jit_hard_reset(
                self.v, self.u, spike_d, self.v_min, self.d, spike
            )

        return self.v

    def neuronal_fire(self):
        """
        Calculate out spikes of neurons by their current membrane potential and v_peak.
        """
        return self.surrogate_function(self.v - self.v_peak)

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", v_min={self.v_min}, v_peak={self.v_peak}, v_r={self.v_r}, u_rest={self.u_rest}, a={self.a}, b={self.b}, d={self.d}"
        )

    def single_step_forward(self, x: torch.Tensor):
        """
        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes and membrane potential of neurons
        :rtype: torch.Tensor
        """
        self.neuronal_charge(x)
        self.neuronal_adaptation()
        spike = self.neuronal_fire()
        v = self.neuronal_reset(spike)
        return spike, v

    def single_step_forward_vt(self, x: torch.Tensor):
        self.neuronal_charge(x)
        self.neuronal_adaptation()
        self.update()
        spike = self.neuronal_fire()
        v = self.neuronal_reset(spike)
        return spike, v

    def update(self):
        pass


class IzhikevichNode(AdaptBaseNode):
    def __init__(
        self,
        n_neuron: int,
        c: Union[float, torch.Tensor],
        v_t: Union[float, torch.Tensor],
        k: Union[float, torch.Tensor],
        v_peak: Union[float, torch.Tensor],
        v_min: Union[float, torch.Tensor],
        v_r: Union[float, torch.Tensor],
        u_rest: Union[float, torch.Tensor],
        a: Union[float, torch.Tensor],
        b: Union[float, torch.Tensor],
        d: Union[float, torch.Tensor],
        surrogate_function: Callable = surrogate.Sigmoid(),
        store_vt: bool = True,
        method="euler",
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        """
        :param c: membrane capacitance
        :type c: float or torch.Tensor

        :param k: a constant that reflects conductance during spike generation
        :type k: float or torch.Tensor

        :param v_r: resting membrane potential
        :type v_r: float or torch.Tensor

        :param v_t: instantaneous threshold potential
        :type v_t: float or torch.Tensor

        :param a: recovery time constant
        :type a: float or torch.Tensor

        :param b: a constant that reflects conductance during repolarization
        :type b: float or torch.Tensor

        :param v_peak: spike cutoff value
        :type v_peak: float or torch.Tensor

        :param d: a constant that reflects the currents activated during a spike
        :type d: float or torch.Tensor

        :param v_min: membrane potential reset value. If not ``None``, the neuron's voltage will be set to ``v_min``
            after firing a spike (hard). If ``None``, the neuron's voltage will subtract ``v_peak`` after firing a spike (soft).
        :type v_min: float or torch.Tensor

        :param store_vt: When using "store_vt = 'Ture'", update U[t] to use V[t-1], otherwise use V[t].
        :type store_vt: bool

        :param method: the iteration method, e.g. 'euler'.
        :type step_mode: str

            The Izhikevich neuron modela, is a kind of nonlinear modela. The subthreshold neural dynamics of it is as follow:

        .. math::
                H[t] = V[t - 1] + \\frac{1}{c}\left(X[t] + k (V[t - 1] - V_{r})(V[t - 1] - v_t)-U[t-1]\\right)
        """
        assert isinstance(c, (float, torch.Tensor)) and (
            c > 1.0 if isinstance(c, float) else torch.all(c > 1.0).item()
        )
        assert isinstance(k, (float, torch.Tensor)) and (
            k > 0.0 if isinstance(k, float) else torch.all(k > 0.0).item()
        )

        assert (
            backend == "torch"
        ), "cupy backend is less tested, torch.compile generally performs better than simply-written cupy"

        super().__init__(
            n_neuron,
            v_min,
            v_peak,
            v_r,
            u_rest,
            a,
            b,
            d,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )
        '''
        # ====================【新增代码部分】====================
        # 在这里，我们用随机化的初始电位覆盖掉父类设置的默认值

        # 1. 计算随机区间的宽度，即 (v_t + 1) - v_r
        range_width = (v_t + 1.0) - v_r
        
        # 2. 生成一个和 v_r 形状相同，数值在 [0, 1) 之间的均匀分布随机张量
        random_scaler = torch.rand_like(v_r)
        
        # 3. 通过缩放和平移，将随机数映射到 [v_r, v_t + 1] 区间
        initial_v_random = v_r + random_scaler * range_width
        
        # 4. 用新生成的随机化电压值更新神经元的初始膜电位 self.v
        self.v = initial_v_random
        # ========================================================
        '''
        #self.v=torch.zeros_like(v_r)
        print("初始化self.v的值是:", self.v)
        
        
        self.register_buffer("c", c.clone().detach(), persistent=False)
        self.register_buffer("v_t", v_t.clone().detach(), persistent=False)
        self.register_buffer("k", k.clone().detach(), persistent=False)
        '''
        self.register_buffer("c", torch.tensor(c), persistent=False)
        self.register_buffer("v_t", torch.tensor(v_t), persistent=False)
        self.register_buffer("k", torch.tensor(k), persistent=False)
        '''
        
        self.store_vt = store_vt
        self.method = method

        if self.method == "euler":
            if self.store_vt is True:
                self.neuronal_charge = self.neuronal_charge_euler_vt
                self.single_step_forward = self.single_step_forward_vt
            else:
                self.neuronal_charge = self.neuronal_charge_euler
                # self.update = lambda: None
            # self.jit_neuronal_adaptation = self.jit_neuronal_adaptation_euler
        elif self.method == "exponential":
            # TODO:
            pass
        else:
            raise ValueError(self.method)

        if isinstance(self.a, float):
            self.multi_step_forward = self.multi_step_forward_float
        elif isinstance(self.a, torch.Tensor):
            self.multi_step_forward = self.multi_step_forward_tensor
        else:
            raise ValueError(type(self.a))

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", c={self.c}, v_t={self.v_t}, k={self.k}, store_vt={self.store_vt}, method={self.method}"
        )

    def neuronal_charge_euler(self, x: torch.Tensor):
        self.v = (
            self.v
            + (x + self.k * (self.v - self.v_r) * (self.v - self.v_t) - self.u) / self.c
        )
        self.v = torch.clamp(self.v, min=-100.0, max=100.0)

    def neuronal_charge_euler_vt(self, x: torch.Tensor):
        self.vt = (
            self.v
            + (x + self.k * (self.v - self.v_r) * (self.v - self.v_t) - self.u) / self.c
        )
        self.vt = torch.clamp(self.vt, min=-100.0, max=100.0)

    def update(self):
        self.v = self.vt

    @property
    def supported_backends(self):
        if self.step_mode == "s":  # torch
            return ("torch",)
        elif self.step_mode == "m":
            return ("torch", "cupy")  # torch and cupy
        else:
            raise ValueError(self.step_mode)

    def multi_step_forward_float(self, x_seq: torch.Tensor):
        if self.backend == "torch":
            return super().multi_step_forward(x_seq)
        elif self.backend == "cupy":
            self.v_float_to_tensor(x_seq[0])
            self.u_float_to_tensor(x_seq[0])

            spike_seq, v_seq, u_seq = (
                neuron_kernel_float.MultiStepIzhikevichNodePTT.apply(
                    x_seq.flatten(1),
                    self.v.flatten(0),
                    self.u.flatten(0),
                    self.c,
                    self.v_peak,
                    self.v_min,
                    self.v_r,
                    self.b,
                    self.d,
                    self.a,
                    self.v_t,
                    self.k,
                    self.detach_reset,
                    self.surrogate_function.cuda_code,
                )
            )

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)
            u_seq = u_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()
            self.u = u_seq[-1].clone()

            return spike_seq, v_seq
        else:
            raise ValueError(self.backend)

    def multi_step_forward_tensor(self, x_seq: torch.Tensor):
        if self.backend == "torch":
            return super().multi_step_forward(x_seq)
        elif self.backend == "cupy":
            self.v_float_to_tensor(x_seq[0])
            self.u_float_to_tensor(x_seq[0])

            spike_seq, v_seq, u_seq = (
                neuron_kernel_tensor.MultiStepIzhikevichNodePTT.apply(
                    x_seq.flatten(1),
                    self.v.flatten(0),
                    self.u.flatten(0),
                    self.c,
                    self.v_peak,
                    self.v_min,
                    self.v_r,
                    self.b,
                    self.d,
                    self.a,
                    self.v_t,
                    self.k,
                    self.detach_reset,
                    self.surrogate_function.cuda_code,
                )
            )

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)
            u_seq = u_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()
            self.u = u_seq[-1].clone()

            #print("当前self.v的值是:", self.v)
            return spike_seq, v_seq
        else:
            raise ValueError(self.backend)
