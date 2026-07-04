主体是事件驱动的prespan算法，spike list相当于任务队列，每个lane采用全局 atomic counter从spikelist领取任务,所以是一个用原子加模拟的隐式队列。

伪代码：

每个lane内：
for t in window_size:
    // Task0: optional external input
    apply_external_input(x_t)
    grid.sync()

    // Task1: LIF update
    // read psc_t + x_t
    // update v
    // emit spike
    // soft reset
    // write spike list
    update_cells_and_emit_spikes()
    grid.sync()

    // Task2: PSC decay
    psc *= exp(-dt / tau_syn)
    grid.sync()

    // Task3: recurrent prespan fanout
    // for each spike (b, pre):
    //     for edge in CSR[pre]:
    //         atomicAdd(psc[b, post], weight)
    process_recurrent_fanout()
    grid.sync()
end.

Task1:update cell，如果发放就填充spikelist队列并复位，领完为止，具体而言，步骤是读 psc，积分，复位，写输出spike
Task2:update psc，用衰减公式，领完为止
Task3:atomic从spikelist领取任务更新电流，直到领取完毕

通过
cudaGetDeviceProperties()
cudaOccupancyMaxActiveBlocksPerMultiprocessor()
动态算出 cooperative kernel 允许的 gridDim

未来正式benchmark数据集参数(目前仅参考)：
N = 4166
nnz = 726404
density = 4.19%
平均出度 = 177.65
中位出度 = 64
P90 = 393
P95 = 726
P99 = 1911
max fanout = 4165

参数设置：
RSNN框架，LIF + ExponentialPSC要求

The first persistent kernel target is intentionally narrow: recurrent RSNN
dynamics with a scalar-parameter LIF neuron and an ``ExponentialPSC`` synapse.
The benchmark and persistent contract share these defaults:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - First-kernel meaning
   * - ``dt``
     - ``1.0``
     - Euler step size for LIF and exponential PSC decay.
   * - ``tau_mem``
     - ``20.0``
     - LIF membrane time constant.
   * - ``tau_syn``
     - ``5.0``
     - Exponential PSC time constant.
   * - ``v_threshold``
     - ``1.0``
     - Spike threshold; spikes are emitted when ``v >= v_threshold``.
   * - ``v_reset``
     - ``0.0``
     - Reset baseline used by the LIF leak and reset delta.
   * - ``c_m``
     - ``1.0``
     - Membrane capacitance divisor for input current.
   * - ``hard_reset``
     - ``False``
     - Soft reset: subtract ``v_threshold - v_reset`` after a spike.
   * - ``window_size``
     - ``128``
     - Suggested processing window length in time steps.
"""
第一版建议明确不支持：
synaptic delay
refractory period
heterogeneous tau_mem
heterogeneous tau_syn
hard reset
multi-compartment neuron
只做：
scalar LIF
scalar ExponentialPSC
soft reset
delay = 0
refractory = none
float32 state / weight
int32 index / counter




容错率要求不严格：
abs error < 1e-5 或 1e-4

