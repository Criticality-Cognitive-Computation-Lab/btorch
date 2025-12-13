import torch
from torch import Tensor

from . import environ
from .base import MemoryModule
from .ode import exp_euler_step_auto


class BasePSC(MemoryModule):
    def __init__(
        self,
        n_neuron: int,
        batch_size: int, # <--- 添加 batch_size 参数
        linear: torch.nn.Module,
        step_mode="s",
        backend="torch",
    ):
        super().__init__()

        self.register_memory("I_psc", 0.0, n_neuron)
        # 创建一个初始值为0的张量来存储突触后电流 I_psc
        initial_psc = torch.zeros(batch_size,n_neuron)
        # 将这个张量注册为名为 'I_psc' 的缓冲区
        self.register_buffer('I_psc', initial_psc)
        
        self.register_memory("h", 0.0, n_neuron)
        # 创建一个初始值为0的张量来存储突触后电流 I_psc
        initial_h = torch.zeros(batch_size,n_neuron)
        # 将这个张量注册为名为 'h' 的缓冲区
        self.register_buffer('h', initial_h)

        self.step_mode = step_mode
        self.backend = backend

        self.linear = linear

    def extra_repr(self):
        return f" step_mode={self.step_mode}, backend={self.backend}"

    def conductance_charge(self):
        raise NotImplementedError()

    def adaptation_charge(self):
        raise NotImplementedError()

    def current_charge(self, v=None):
        if v is not None:
            raise NotImplementedError(
                "Only current-based I_psc is supported."
                "Conductance-based I_psc requires voltage from post-syn neurons, "
                "which the current abstraction doesn't support."
            )
        else:
            return self.I_psc

    def single_step_forward(self, z: torch.Tensor):
        self.conductance_charge()
        self.adaptation_charge(z)
        current = self.current_charge()
        return current

    def multi_step_forward(self, z_seq: torch.Tensor):
        T = z_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(z_seq[t])
            y_seq.append(y)

        return torch.stack(y_seq)


class AlphaPSCBilleh(BasePSC):
    def __init__(
        self,
        n_neuron: int,
        batch_size: int, # <--- 添加 batch_size 参数
        tau_syn: float | torch.Tensor,
        linear: torch.nn.Module,
        step_mode="s",
        backend="torch",
    ):
        """The Current-Based Alpha form of I_psc, from [1], ensuring a post-
        synaptic current with synapse weight W = 1.0 has an amplitude of 1.0 pA
        at the peak time point of t = tau_syn.

        NOTE: it does **NOT** respect environ.get("dt")

        [1] Billeh, Y. N. et al. Systematic integration of structural and
        functional data into multi-scale models of mouse primary visual
        cortex. 662189 Preprint at https://doi.org/10.1101/662189 (2019).

        :param tau_syn: the synaptic time constant
        :type tau_syn: float or torch.Tensor
        """

        super().__init__(n_neuron,batch_size, linear, step_mode, backend)

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)

        self.register_buffer(
            "syn_decay", torch.exp(-1.0 / self.tau_syn), persistent=False
        )

    def conductance_charge(self):
        self.I_psc = self.syn_decay * self.I_psc + self.syn_decay * self.h
        return self.I_psc

    def adaptation_charge(self, z: torch.Tensor):
        wz = self.linear(z)
        self.h = self.syn_decay * self.h + torch.e / self.tau_syn * wz


class AlphaPSC(BasePSC):
    def __init__(
        self,
        n_neuron: int,
        tau_syn,
        linear: torch.nn.Module,
        g_max=1.0,
        step_mode="s",
        backend="torch",
    ):
        """The Alpha form (current-based) of I_psc, from Brainpy/BrainState."""

        super().__init__(n_neuron, linear, step_mode, backend)

        self.register_buffer("tau_syn", torch.as_tensor(tau_syn), persistent=False)
        self.register_buffer("g_max", torch.as_tensor(g_max), persistent=False)

    def dg(self, I_psc, h):
        return -I_psc / self.tau_syn + h / self.tau_syn

    def dh(self, h):
        return -h / self.tau_syn

    def conductance_charge(self):
        self.I_psc = exp_euler_step_auto(self.dg, self.I_psc, self.h, dt=environ.get("dt"))

    def adaptation_charge(self, z: Tensor):
        wz = self.g_max * self.linear(z)
        self.h = exp_euler_step_auto(self.dh, self.h, dt=environ.get("dt")) + wz

'''
##这个类是在/home/maosiyu/Project_01/5/Model_test1/test_dynamics_xlsx.py中用于双神经元突触测试的,会打印出来很多信息
class HeterogeneousAlphaPSC(BasePSC):
    """
    An Alpha PSC model that supports different time constants for
    E-E, I-E, E-I, and I-I connections.
    
    *** NOTE: This version contains detailed print statements for debugging
    the effect of a presynaptic spike on a postsynaptic neuron. ***
    """
    def __init__(
        self,
        n_neuron: int,
        batch_size: int,
        linear: torch.nn.Module,
        e_i_mask: torch.Tensor,  # A 1D boolean tensor, True for excitatory neurons
        tau_map: dict,
        step_mode="s",
        backend="torch",
        monitor=None,
    ):
        super().__init__(n_neuron, batch_size, linear, step_mode, backend)
        
        # --- Added for debugging/logging ---
        self.time_step = 0
        self._spike_detected_in_step = False
        
        self.monitor = monitor
        
        self.register_buffer("e_i_mask", e_i_mask)
        e_i_mask_float = self.e_i_mask.float().unsqueeze(0)

        tau_e_post = e_i_mask_float * tau_map['E-E'] + (1 - e_i_mask_float) * tau_map['E-I']
        tau_i_post = e_i_mask_float * tau_map['I-E'] + (1 - e_i_mask_float) * tau_map['I-I']
        
        self.register_buffer("syn_decay_e", torch.exp(-1.0 / tau_e_post), persistent=False)
        self.register_buffer("syn_decay_i", torch.exp(-1.0 / tau_i_post), persistent=False)
        self.register_buffer("scale_e", torch.e / tau_e_post, persistent=False)
        self.register_buffer("scale_i", torch.e / tau_i_post, persistent=False)

        self.register_buffer('I_psc_e', torch.zeros(batch_size, n_neuron))
        self.register_buffer('h_e', torch.zeros(batch_size, n_neuron))
        self.register_buffer('I_psc_i', torch.zeros(batch_size, n_neuron))
        self.register_buffer('h_i', torch.zeros(batch_size, n_neuron))

    
    def conductance_charge(self):
        """Updates the postsynaptic currents based on their decay rates."""
        # For logging, we focus on the postsynaptic neuron (index 1) from your test script.
        post_syn_idx = 1
        
        I_psc_e_before = self.I_psc_e[0, post_syn_idx].item()
        h_e_current = self.h_e[0, post_syn_idx].item()

        # --- Original Update Logic ---
        self.I_psc_e = self.syn_decay_e * self.I_psc_e.detach() + self.syn_decay_e * self.h_e
        self.I_psc_i = self.syn_decay_i * self.I_psc_i.detach() + self.syn_decay_i * self.h_i

        # --- Logging Logic ---
        # If a spike was detected in the adaptation step, print the details of this step.
        if self._spike_detected_in_step:
            decay_e = self.syn_decay_e[0, post_syn_idx].item()
            I_psc_e_after = self.I_psc_e[0, post_syn_idx].item()
            print(f"  └─[Step 2: Conductance Update] for Post-Syn Neuron {post_syn_idx}:")
            print(f"    - I_psc_e was: {I_psc_e_before:.4f}")
            print(f"    - Update Rule: I_psc_e_new = decay * I_psc_e_old + decay * h_e_new")
            print(f"    - Calculation: {I_psc_e_after:.4f} = {decay_e:.4f} * {I_psc_e_before:.4f} + {decay_e:.4f} * {h_e_current:.4f}")
            print(f"    - I_psc_e is now: {I_psc_e_after:.4f}")
            print("="*70)
            
    def adaptation_charge(self, z: torch.Tensor):
        """Updates the intermediate 'h' state based on incoming spikes 'z'."""
        self.time_step += 1
        self._spike_detected_in_step = False
        
        # For logging, focus on the postsynaptic neuron (index 1).
        post_syn_idx = 1
        h_e_before = self.h_e[0, post_syn_idx].item()

        # --- Original Update Logic ---
        z_e = z * self.e_i_mask
        z_i = z * ~self.e_i_mask
        wz_e = self.linear(z_e)
        wz_i = self.linear(z_i)
        self.h_e = self.syn_decay_e * self.h_e.detach() + self.scale_e * wz_e
        self.h_i = self.syn_decay_i * self.h_i.detach() + self.scale_i * wz_i

        # --- Logging Logic ---
        # Check if any neuron spiked in this time step
        spiking_indices = (z > 0).nonzero(as_tuple=True)[1]
        if len(spiking_indices) > 0:
            pre_syn_idx = spiking_indices[0].item()
            self._spike_detected_in_step = True

            print(f"\n" + "="*70)
            print(f"T = {self.time_step}: Neuron {pre_syn_idx} Fired! Analyzing effect on Neuron {post_syn_idx}.")
            print(f"----------------------------------------------------------------------")
            
            wz_e_effect = wz_e[0, post_syn_idx].item()
            scale_e = self.scale_e[0, post_syn_idx].item()
            decay_e = self.syn_decay_e[0, post_syn_idx].item()
            h_e_after = self.h_e[0, post_syn_idx].item()

            print(f"  ┌─[Step 1: Adaptation Update] for Post-Syn Neuron {post_syn_idx}:")
            print(f"    - h_e before update: {h_e_before:.4f}")
            print(f"    - Incoming weighted spike (wz_e) from Neuron {pre_syn_idx}: {wz_e_effect:.4f}")
            print(f"    - Update Rule: h_e_new = (decay * h_e_old) + (scale * wz_e)")
            print(f"    - Decay Part: ({decay_e:.4f} * {h_e_before:.4f}) = {decay_e * h_e_before:.4f}")
            print(f"    - Spike Part: ({scale_e:.4f} * {wz_e_effect:.4f}) = {scale_e * wz_e_effect:.4f}")
            print(f"    - h_e after update: {h_e_after:.4f}")
    
    def current_charge(self, v=None):
        """Returns the total current, which is the sum of E and I components."""
        # 计算总电流
        total_current = self.I_psc_e + self.I_psc_i
        # ▼▼▼ 核心修正：用计算出的总电流更新 self.I_psc 变量 ▼▼▼
        self.I_psc = total_current
        return self.I_psc
        #return self.I_psc_e + self.I_psc_i
        
    def single_step_forward(self, z: torch.Tensor):
        """
        Performs a single forward step for one time-step.
        The order is important: first process incoming spikes, then decay the currents.
        """
        self.adaptation_charge(z)
        self.conductance_charge()
        current = self.current_charge()
        
        if self.monitor is not None:
            self.monitor.record(self.I_psc_e, self.I_psc_i)
            
        return current
'''

# synapse.py (请用这个版本替换文件中的 HeterogeneousAlphaPSC 类)
'''
class HeterogeneousAlphaPSC(BasePSC):
    """
    An Alpha PSC model that supports different time constants for
    E-E, I-E, E-I, and I-I connections.
    【修正版：已移除 .detach() 调用以支持BPTT训练】
    """
    def __init__(
        self,
        n_neuron: int,
        batch_size: int,
        linear: torch.nn.Module,
        e_i_mask: torch.Tensor,
        tau_map: dict,
        step_mode="s",
        backend="torch"
        # monitor 参数可以根据需要添加回来
    ):
        super().__init__(n_neuron, batch_size, linear, step_mode, backend)
        
        self.register_buffer("e_i_mask", e_i_mask)
        e_i_mask_float = self.e_i_mask.float().unsqueeze(0)

        tau_e_post = e_i_mask_float * tau_map['E-E'] + (1 - e_i_mask_float) * tau_map['E-I']
        tau_i_post = e_i_mask_float * tau_map['I-E'] + (1 - e_i_mask_float) * tau_map['I-I']
        
        self.register_buffer("syn_decay_e", torch.exp(-1.0 / tau_e_post), persistent=False)
        self.register_buffer("syn_decay_i", torch.exp(-1.0 / tau_i_post), persistent=False)
        self.register_buffer("scale_e", torch.e / tau_e_post, persistent=False)
        self.register_buffer("scale_i", torch.e / tau_i_post, persistent=False)

        self.register_buffer('I_psc_e', torch.zeros(batch_size, n_neuron))
        self.register_buffer('h_e', torch.zeros(batch_size, n_neuron))
        self.register_buffer('I_psc_i', torch.zeros(batch_size, n_neuron))
        self.register_buffer('h_i', torch.zeros(batch_size, n_neuron))

    
    def conductance_charge(self):
        """更新突触后电流 (已移除 .detach())"""
        self.I_psc_e = self.syn_decay_e * self.I_psc_e + self.syn_decay_e * self.h_e
        self.I_psc_i = self.syn_decay_i * self.I_psc_i + self.syn_decay_i * self.h_i

    def adaptation_charge(self, z: torch.Tensor):
        """根据传入的脉冲更新中间变量 'h' (已移除 .detach())"""
        z_e = z * self.e_i_mask
        z_i = z * ~self.e_i_mask

        wz_e = self.linear(z_e)
        wz_i = self.linear(z_i)

        self.h_e = self.syn_decay_e * self.h_e + self.scale_e * wz_e
        self.h_i = self.syn_decay_i * self.h_i + self.scale_i * wz_i
    
    def current_charge(self, v=None):
        total_current = self.I_psc_e + self.I_psc_i
        # 以下两行用于兼容旧的记录方式，可以保留
        self.I_psc = total_current 
        return self.I_psc
        
    def single_step_forward(self, z: torch.Tensor):
        self.adaptation_charge(z)
        self.conductance_charge()
        current = self.current_charge()
        return current
    # synapse.py (在 HeterogeneousAlphaPSC 类中，用这个新版本替换旧的 reset 方法)

    def reset(self):
        """
        【最终修正版】重置所有内部状态张量。
        使用 x = x.detach() 代替 x.detach_() 以处理张量视图。
        """
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心修改处 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 使用 x = x.detach() 的方式来分离张量
        self.I_psc_e = self.I_psc_e.detach()
        self.I_psc_i = self.I_psc_i.detach()
        self.h_e = self.h_e.detach()
        self.h_i = self.h_i.detach()
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 就地清零操作保持不变
        self.I_psc_e.zero_()
        self.I_psc_i.zero_()
        self.h_e.zero_()
        self.h_i.zero_()
        
        print("--- [DEBUG] HeterogeneousAlphaPSC.reset() has been called. ---")
'''       
# Add this new class to your project.
# It inherits from your provided BasePSC and replaces AlphaPSCBilleh.

class HeterogeneousAlphaPSC(BasePSC):
    """
    An Alpha PSC model that supports different time constants for
    E-E, I-E, E-I, and I-I connections.
    """
    def __init__(
        self,
        n_neuron: int,
        batch_size: int,
        linear: torch.nn.Module,
        e_i_mask: torch.Tensor,  # A 1D boolean tensor, True for excitatory neurons
        tau_map: dict,
        step_mode="s",
        backend="torch"
    ):
        # Call the parent constructor
        super().__init__(n_neuron, batch_size, linear, step_mode, backend)
        
        # Register the E/I mask. This tells the class which neurons are excitatory
        # and which are inhibitory. It's crucial for all subsequent calculations.
        self.register_buffer("e_i_mask", e_i_mask)
        e_i_mask_float = self.e_i_mask.float().unsqueeze(0)  # Shape: [1, n_neuron]

        # --- Create tau and decay tensors based on POST-synaptic neuron type ---
        
        # Tau vector for inputs originating FROM excitatory neurons.
        # If the POST-synaptic neuron is E, use tau_map['E-E'].
        # If the POST-synaptic neuron is I, use tau_map['E-I'].
        tau_e_post = e_i_mask_float * tau_map['E-E'] + (1 - e_i_mask_float) * tau_map['E-I']
        
        # Tau vector for inputs originating FROM inhibitory neurons.
        # If the POST-synaptic neuron is E, use tau_map['I-E'].
        # If the POST-synaptic neuron is I, use tau_map['I-I'].
        tau_i_post = e_i_mask_float * tau_map['I-E'] + (1 - e_i_mask_float) * tau_map['I-I']
        
        # Register decay and scaling factors as buffers so they are managed by PyTorch
        self.register_buffer("syn_decay_e", torch.exp(-1.0 / tau_e_post), persistent=False)
        self.register_buffer("syn_decay_i", torch.exp(-1.0 / tau_i_post), persistent=False)
        self.register_buffer("scale_e", torch.e / tau_e_post, persistent=False)
        self.register_buffer("scale_i", torch.e / tau_i_post, persistent=False)

        # --- Register separate state variables for E and I inputs ---
        # Note: We directly initialize buffers to match the expected statefulness.
        self.register_buffer('I_psc_e', torch.zeros(batch_size, n_neuron))
        self.register_buffer('h_e', torch.zeros(batch_size, n_neuron))
        self.register_buffer('I_psc_i', torch.zeros(batch_size, n_neuron))
        self.register_buffer('h_i', torch.zeros(batch_size, n_neuron))

    
    def conductance_charge(self):
        """Updates the postsynaptic currents based on their decay rates."""
        # I_psc 的更新依赖于上一时刻的 I_psc 和 h, 所以将上一时刻的 detach
        self.I_psc_e = self.syn_decay_e * self.I_psc_e.detach() + self.syn_decay_e * self.h_e
        self.I_psc_i = self.syn_decay_i * self.I_psc_i.detach() + self.syn_decay_i * self.h_i

    def adaptation_charge(self, z: torch.Tensor):
        """Updates the intermediate 'h' state based on incoming spikes 'z'."""
        z_e = z * self.e_i_mask
        z_i = z * ~self.e_i_mask

        wz_e = self.linear(z_e)
        wz_i = self.linear(z_i)

        # h 的更新依赖于上一时刻的 h 和当前输入 wz, 将上一时刻的 h detach
        self.h_e = self.syn_decay_e * self.h_e.detach() + self.scale_e * wz_e
        self.h_i = self.syn_decay_i * self.h_i.detach() + self.scale_i * wz_i
    '''
    def conductance_charge(self):
        """Updates the postsynaptic currents based on their decay rates."""
        self.I_psc_e = self.syn_decay_e * self.I_psc_e + self.syn_decay_e * self.h_e
        self.I_psc_i = self.syn_decay_i * self.I_psc_i + self.syn_decay_i * self.h_i

    def adaptation_charge(self, z: torch.Tensor):
        """Updates the intermediate 'h' state based on incoming spikes 'z'."""
        # z has shape [batch_size, n_neurons]
        
        # 1. Separate PRE-synaptic spikes into excitatory and inhibitory groups
        z_e = z * self.e_i_mask
        z_i = z * ~self.e_i_mask

        # 2. Calculate the post-synaptic effect from each group
        wz_e = self.linear(z_e)
        wz_i = self.linear(z_i)

        # 3. Update the h-states with their respective inputs and decays
        self.h_e = self.syn_decay_e * self.h_e + self.scale_e * wz_e
        self.h_i = self.syn_decay_i * self.h_i + self.scale_i * wz_i
    '''
    
    def current_charge(self, v=None):
        """Returns the total current, which is the sum of E and I components."""
        return self.I_psc_e + self.I_psc_i
        
    def single_step_forward(self, z: torch.Tensor):
        """
        Performs a single forward step for one time-step.
        The order is important: first process incoming spikes, then decay the currents.
        """
        self.adaptation_charge(z)
        self.conductance_charge()
        current = self.current_charge()

        #from . import environ # 可能需要引入environ来获取当前时间步
        #if environ.get_t() % 10 == 0:
        '''
        print(
            f"[Synapse DEBUG] Raw PSC -> "
            f"E_mean: {self.I_psc_e.mean():.4f}, E_max: {self.I_psc_e.max():.4f} | "
            f"I_mean: {self.I_psc_i.mean():.4f}, I_max: {self.I_psc_i.max():.4f}"
        )
        '''
        
        #if self.I_psc_e.mean() > 0.1: # 只在有活动时打印
            #print(f"[DEBUG PSC] I_psc_e Mean: {self.I_psc_e.mean().item():.2f}, Max: {self.I_psc_e.max().item():.2f}")
        return current