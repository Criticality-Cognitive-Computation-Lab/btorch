import numpy as np
import torch
from scipy.stats import kurtosis, skew


def calculate_fr_distribution(spikes, dt=1.0):
    """计算群体中每个时刻的平均发放率，并统计其分布特征。

    Args:
        spikes: (Time, Neurons) 脉冲矩阵
        dt: 仿真步长(ms)
    Returns:
        dict: {'rates': array, 'mean': float, 'skew': float, 'kurt': float}
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()

    # 计算每个时间点的平均发放率 (Hz)
    window_size = 5
    kernel = np.ones(window_size) / (window_size * dt / 1000.0)  # 转换为Hz
    pop_spikes = spikes.mean(axis=1)  # (T,)
    rates = np.convolve(pop_spikes, kernel, mode="same")

    return {
        "rates": rates,  # 每个神经元的发放率分布
        "mean": np.mean(rates),  # 均值
        "skew": skew(rates),  # 偏度
        "kurt": kurtosis(rates),  # 峰度
    }


def calculate_cv_isi(spikes, dt=1.0):
    """计算群体中每个神经元的CV_ISI，并统计其分布特征。

    Args:
        spikes: (Time, Neurons) 脉冲矩阵
        dt: 仿真步长(ms)
    Returns:
        dict: {'cv_isi': array, 'mean': float}
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()

    num_neurons = spikes.shape[1]
    cv_isi_list = []

    for n in range(num_neurons):
        spike_times = np.where(spikes[:, n] > 0)[0] * dt  # 转换为ms
        if len(spike_times) < 2:
            cv_isi_list.append(np.nan)  # 不足两个脉冲，无法计算ISI
            continue

        isis = np.diff(spike_times)  # 计算ISI
        if np.mean(isis) == 0:
            cv_isi_list.append(np.nan)
            continue

        cv_isi = np.std(isis) / np.mean(isis)
        cv_isi_list.append(cv_isi)

    cv_isi_array = np.array(cv_isi_list)
    mean_cv_isi = np.nanmean(cv_isi_array)  # 忽略NaN值计算均值

    return {
        "cv_isi": cv_isi_array,  # 每个神经元的CV_ISI分布
        "mean": mean_cv_isi,  # 均值
    }


def calculate_spike_distance(spikes, dt=1.0, subset_size=100, seed=None):
    """计算 SPIKE-distance (Kreuz et al., 2013)。 衡量脉冲序列之间的不同步程度。0表示完全同步。

    Args:
        spikes: (Time, Neurons) 脉冲矩阵
        dt: 仿真步长(ms)
        subset_size: 随机抽样的神经元数量，用于计算成对距离
    Returns:
        float: 平均 SPIKE-distance
    """
    if isinstance(spikes, torch.Tensor):
        spikes = spikes.detach().cpu().numpy()

    T_steps, N = spikes.shape
    times = np.arange(T_steps) * dt

    # 随机抽样
    if N > subset_size:
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(N, subset_size, replace=False)
        selected_spikes = spikes[:, indices]
        N_subset = subset_size
    else:
        selected_spikes = spikes
        N_subset = N

    # 预计算每个神经元的 t_prev, t_next, isi
    # shape: (N_subset, T_steps)
    t_prev = np.zeros((N_subset, T_steps))
    t_next = np.zeros((N_subset, T_steps))
    isi = np.zeros((N_subset, T_steps))

    for n in range(N_subset):
        spike_indices = np.where(selected_spikes[:, n] > 0)[0]
        spike_times = spike_indices * dt

        if len(spike_times) == 0:
            # 处理无脉冲情况：设为无穷大或整个区间
            t_prev[n, :] = 0
            t_next[n, :] = times[-1]
            isi[n, :] = times[-1]
            continue

        # 使用 searchsorted 找到每个时间点的前后脉冲
        # indices_next 指向 times 中每个 t 之后的第一个脉冲在 spike_times 中的位置
        indices_next = np.searchsorted(spike_times, times)

        # 处理边界
        indices_next = np.clip(indices_next, 0, len(spike_times) - 1)
        # indices_prev = np.clip(indices_next - 1, 0, len(spike_times) - 1)

        # 修正 searchsorted 的结果，确保 t_prev <= t <= t_next
        # 对于 t 正好在 spike_time 上的情况，searchsorted 可能返回当前或下一个
        # 这里我们简单处理：
        # t_next[t] 是 >= t 的第一个脉冲
        # t_prev[t] 是 <= t 的最后一个脉冲

        # 更精确的做法：
        # t_prev: max(s | s <= t)
        # t_next: min(s | s > t)  (SPIKE-distance 定义通常要求严格大于，或者 >=)

        # 重新实现简单的循环填充（虽然慢一点但准确）或者利用 searchsorted 的性质
        # 实际上，对于 step function，可以用 diff 填充

        # 快速填充法：
        # t_prev
        curr_spike = 0.0
        spike_idx = 0
        for t_idx, t in enumerate(times):
            if spike_idx < len(spike_times) and t >= spike_times[spike_idx]:
                curr_spike = spike_times[spike_idx]
                # 如果不是最后一个脉冲，检查是否到了下一个
                if spike_idx < len(spike_times) - 1 and t >= spike_times[spike_idx + 1]:
                    spike_idx += 1
                    curr_spike = spike_times[spike_idx]
            t_prev[n, t_idx] = curr_spike

        # t_next
        curr_spike = times[-1]
        spike_idx = len(spike_times) - 1
        for t_idx in range(T_steps - 1, -1, -1):
            t = times[t_idx]
            if spike_idx >= 0 and t <= spike_times[spike_idx]:
                curr_spike = spike_times[spike_idx]
                if spike_idx > 0 and t <= spike_times[spike_idx - 1]:
                    spike_idx -= 1
                    curr_spike = spike_times[spike_idx]
            t_next[n, t_idx] = curr_spike

        isi[n, :] = t_next[n, :] - t_prev[n, :]

        isi[n, isi[n, :] == 0] = dt

    # 计算成对 SPIKE-distance S(t) = ( |dt_p1 - dt_p2| * isi2 + |dt_f1 - dt_f2|
    # * isi1 ) / ( 0.5 * (isi1 + isi2)**2 )

    dt_p = times[None, :] - t_prev  # (N, T)
    dt_f = t_next - times[None, :]  # (N, T)

    pairwise_distances = []

    # 随机选取若干对进行计算，或者计算所有对如果 N_subset 很大，计算所有对可能较
    # 慢。这里 N_subset 默认为 50， 50*49/2 = 1225，可以接受。
    for i in range(N_subset):
        for j in range(i + 1, N_subset):
            isi1 = isi[i]
            isi2 = isi[j]

            avg_isi_sq = 0.5 * (isi1 + isi2) ** 2
            # 避免除以零
            avg_isi_sq[avg_isi_sq == 0] = 1.0

            term1 = np.abs(dt_p[i] - dt_p[j]) * isi2
            term2 = np.abs(dt_f[i] - dt_f[j]) * isi1

            s_t = (term1 + term2) / avg_isi_sq

            # 时间积分（平均）
            dist = np.mean(s_t)
            pairwise_distances.append(dist)

    if not pairwise_distances:
        return 0.0

    return np.mean(pairwise_distances)
