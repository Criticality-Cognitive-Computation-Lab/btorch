# analysis.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from scipy import signal
import pandas as pd
import os 
'''
def plot_sweep_raster(spikes_tensor, modified_areas, g, strength):
    """
    为参数扫描中的单次运行绘制并保存脉冲活动图。
    """
    print(f"--- 正在为 g={g}, strength={strength} 绘制脉冲活动图... ---")
    
    spikes_np = spikes_tensor.cpu().numpy()
    # 使用批次中的第一个样本进行绘图
    sample_spikes = spikes_np[:, 0, :]
    
    regions_to_plot = ['EC', 'DG', 'CA3', 'CA1', 'SUB']
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    fig, axes = plt.subplots(len(regions_to_plot), 1, figsize=(50, 50), sharex=True)
    fig.suptitle(f'Spike Activity for g={g}, strength={strength}', fontsize=16)

    for i, region_name in enumerate(regions_to_plot):
        ax = axes[i]
        # 创建一个布尔掩码来选择属于当前区域的神经元
        mask = np.array([region_name in area for area in modified_areas])
        region_spikes_sample = sample_spikes[:, mask]
        
        # 找到脉冲的时间和神经元索引
        time, neuron_idx = np.where(region_spikes_sample > 0)
        
        if len(time) > 0:
            ax.scatter(time, neuron_idx, s=3, c=colors[i], label=f'{region_name} Spikes: {len(time)}')
        
        ax.set_title(f"{region_name} spike activity")
        ax.set_ylabel(f"{region_name} neuron index")
        ax.set_xlim(0, sample_spikes.shape[0])
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
    ax.set_xlabel("Time Step")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # 创建一个专门用于存放光栅图的目录
    output_dir = 'raster_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.join(output_dir, f"raster_g_{g}_strength_{strength}.png")
    plt.savefig(filename)
    plt.close(fig) # 关闭图像以释放内存
    print(f"--- 脉冲图已保存至 {filename} ---")
'''

# analysis.py

def plot_sweep_raster(spikes_tensor, modified_areas, g, strength, output_filename=None):
    """
    【诊断增强版】：在光栅图下方增加了网络全局及各区域的瞬时平均发放率曲线图。
    """
    print("modified_areas:",modified_areas)
    print(f"--- 正在为 g={g}, strength={strength} 绘制脉冲活动图... ---")
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 核心修改处 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 在调用 .cpu().numpy() 之前，先调用 .detach()
    spikes_np = spikes_tensor.detach().cpu().numpy()
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 修改结束 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    sample_spikes = spikes_np[:, 0, :]
    
    regions_to_plot = ['EC', 'DG', 'CA3', 'CA1', 'SUB']
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    num_subplots = len(regions_to_plot) + 1
    fig, axes = plt.subplots(num_subplots, 1, figsize=(80, 65), sharex=True, 
                             gridspec_kw={'height_ratios': [3] * len(regions_to_plot) + [2]})
    fig.suptitle(f'Spike Activity and Firing Rate for g={g}, strength={strength}', fontsize=60)

    '''
    # --- 绘制光栅图 (不变) ---
    for i, region_name in enumerate(regions_to_plot):
        ax = axes[i]
        mask = np.array([region_name in area for area in modified_areas])
        region_spikes_sample = sample_spikes[:, mask]
        time, neuron_idx = np.where(region_spikes_sample > 0)
        if len(time) > 0:
            ax.scatter(time, neuron_idx, s=3, c=colors[i], label=f'{region_name} Spikes: {len(time)}')
        ax.set_title(f"{region_name} spike activity", fontsize=40)
        ax.set_ylabel(f"{region_name} neuron index", fontsize=30)
        ax.legend(loc='upper right', fontsize=25)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=20)
    '''
    
    # --- 2. 核心修改：在循环中直接使用 modified_areas 进行 E/I 分离 ---
    for i, region_name in enumerate(regions_to_plot):
        ax = axes[i]
        
        # a. 找到属于当前区域的所有神经元的全局索引
        region_global_indices = np.where(np.core.defchararray.find(modified_areas, region_name) != -1)[0]

        # b. 在这些神经元中，根据后缀 '_pos' 和 '_neg' 区分 E 和 I
        #    我们先获取这些神经元对应的 area 字符串
        region_area_strings = modified_areas[region_global_indices]
        
        #    然后创建布尔掩码
        e_mask_in_region = np.core.defchararray.endswith(region_area_strings, '_pos')
        i_mask_in_region = np.core.defchararray.endswith(region_area_strings, '_neg')
        
        #    最后得到 E 和 I 神经元的全局索引
        e_neuron_global_indices = region_global_indices[e_mask_in_region]
        i_neuron_global_indices = region_global_indices[i_mask_in_region]

        # c. 分别绘制兴奋性(E)神经元的脉冲 (在Y轴正半轴)
        e_spikes = sample_spikes[:, e_neuron_global_indices]
        time_e, neuron_idx_e = np.where(e_spikes > 0)
        if len(time_e) > 0:
            ax.scatter(time_e, neuron_idx_e, s=3, c=colors[i], label=f'E Spikes: {len(time_e)}')

        # d. 分别绘制抑制性(I)神经元的脉冲 (在Y轴负半轴)
        i_spikes = sample_spikes[:, i_neuron_global_indices]
        time_i, neuron_idx_i = np.where(i_spikes > 0)
        if len(time_i) > 0:
            ax.scatter(time_i, -neuron_idx_i - 1, s=3, c=colors[i], marker='|', label=f'I Spikes: {len(time_i)}')
            
        # e. 画分割线并调整Y轴
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.set_title(f"{region_name} spike activity (E on top, I on bottom)", fontsize=40)
        ax.set_ylabel("Neuron Index", fontsize=30)
        ax.legend(loc='upper right', fontsize=25)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        
    # --- 计算并绘制发放率 (核心修改部分) ---
    rate_ax = axes[-1]
    time_steps = np.arange(sample_spikes.shape[0])
    
    # 1. 计算并绘制全局发放率 (黑色粗线)
    instantaneous_firing_rate_hz = np.mean(sample_spikes, axis=1) * 1000
    rate_ax.plot(time_steps, instantaneous_firing_rate_hz, color='black', linewidth=2.0, label='Global Mean Rate')
    
    # --- 2. 新增：循环计算并绘制每个脑区的发放率 (彩色细线) ---
    for i, region_name in enumerate(regions_to_plot):
        mask = np.array([region_name in area for area in modified_areas])
        # 如果某个区域没有神经元，则跳过
        if not np.any(mask):
            continue
        region_spikes_sample = sample_spikes[:, mask]
        # 计算该区域自己的平均发放率
        region_rate_hz = np.mean(region_spikes_sample, axis=1) * 1000
        rate_ax.plot(time_steps, region_rate_hz, color=colors[i], linewidth=0.8, alpha=0.9, label=f'{region_name} Rate')
        
    rate_ax.set_title("Network-wide and Regional Instantaneous Firing Rate", fontsize=40)
    rate_ax.set_ylabel("Firing Rate (Hz)", fontsize=30)
    rate_ax.grid(True, linestyle='--', alpha=0.6)
    rate_ax.legend(loc='upper right', fontsize=25)
    rate_ax.tick_params(axis='both', which='major', labelsize=20)
    
    axes[-1].set_xlim(0, sample_spikes.shape[0])
    axes[-1].set_xlabel("Time Step", fontsize=30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    
    # ==========================================================
    #      【核心修改】替换掉固定的文件保存逻辑
    # ==========================================================
    if output_filename:
        # 如果调用时提供了 output_filename，就保存到指定路径
        # 确保目标目录存在
        output_dir = os.path.dirname(output_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_filename)
        print(f"--- 脉冲图已保存至 {output_filename} ---") # 这条信息可以由调用者打印
    else:
        # 如果没有提供，就像以前一样直接显示图像
        print("--- 未提供输出文件名，将直接显示图像 ---")
        plt.show()
    
    # 最好在保存或显示后关闭图形，以释放内存
    plt.close(fig)
    # ==========================================================


def calculate_cv(spikes_tensor):
    """从脉冲张量计算每个神经元的CV值。"""
    if isinstance(spikes_tensor, torch.Tensor):
        spikes_tensor = spikes_tensor.cpu().numpy()
    T, N = spikes_tensor.shape
    cv_values = np.full(N, np.nan)
    for neuron_idx in range(N):
        spike_times = np.where(spikes_tensor[:, neuron_idx] > 0)[0]
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            if np.mean(isis) > 0:
                cv_values[neuron_idx] = np.std(isis) / np.mean(isis)
    return cv_values

def calculate_fano_factor(all_spike_counts):
    """计算法诺因子(Fano Factor)。"""
    mean_counts = np.mean(all_spike_counts, axis=0)
    var_counts = np.var(all_spike_counts, axis=0)
    fano_factors = np.divide(var_counts, mean_counts, out=np.full_like(mean_counts, np.nan), where=mean_counts != 0)
    return fano_factors

def plot_power_spectrum(spikes_tensor, dt=1e-3):
    """计算并绘制网络全局活动的功率谱密度（PSD）图。"""
    print("\n--- 正在生成功率谱密度 (PSD) 分析图 ---")
    global_activity = torch.sum(spikes_tensor[:, 0, :], dim=1).cpu().numpy()
    fs = 1 / dt
    f, Pxx = signal.welch(global_activity, fs, nperseg=1024)
    plt.figure(figsize=(10, 8))
    plt.loglog(f, Pxx, label='Original power spectrum', color='blue', alpha=0.8)
    fit_range = (f >= 1) & (f <= 100)
    if np.any(fit_range):
        log_f, log_Pxx = np.log10(f[fit_range]), np.log10(Pxx[fit_range])
        slope, intercept = np.polyfit(log_f, log_Pxx, 1)
        beta = -slope
        print(f"✅ 在 1-100Hz 范围内拟合出的幂律指数 (β) ≈ {beta:.2f}")
        fit_line = 10**(slope * np.log10(f) + intercept)
        plt.loglog(f, fit_line, 'r--', label=f'1/f^{beta:.2f} Fitted line')
    plt.title('Power Spectral Density of Network Global Activities')
    plt.xlabel('Frequency (Hz) [Logarithmic scale]')
    plt.ylabel('Power/Hz [Logarithmic scale]')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.savefig("power_spectrum_diagnosis.png")
    plt.show()
        
def visualize_activity_and_weights(model, data_loader, device, modified_areas):
    """可视化模型单次前向传播后的脉冲活动和循环层权重分布。"""
    print("\n--- 开始生成可视化诊断图 ---")
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(data_loader))
        data = data.permute(4, 0, 1, 2, 3).to(device)
        _, spikes = model(data)
        model.reset_state()

    # 1. Spike Raster Plot
    print("正在绘制脉冲活动图...")
    spikes_np = spikes.cpu().numpy()
    sample_spikes = spikes_np[:, 0, :]
    regions_to_plot = ['EC', 'DG', 'CA3', 'CA1', 'SUB']
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    plt.figure(figsize=(18, 20))
    for i, region_name in enumerate(regions_to_plot):
        ax = plt.subplot(len(regions_to_plot), 1, i + 1)
        mask = np.array([region_name in area for area in modified_areas])
        region_spikes_sample = sample_spikes[:, mask]
        time, neuron_idx = np.where(region_spikes_sample > 0)
        if len(time) > 0:
            ax.scatter(time, neuron_idx, s=4, c=colors[i], label=f'{region_name} Spikes: {len(time)}')
        ax.set_title(f"{region_name} spike activity")
        ax.set_ylabel(f"{region_name} neuron index")
        ax.set_xlim(0, sample_spikes.shape[0])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == len(regions_to_plot) - 1:
            ax.set_xlabel("Time Step")
    plt.tight_layout()
    plt.savefig("spike_activity_diagnosis.png")
    plt.show()

    # 2. Weight Distribution
    print("正在绘制循环层权重分布图...")
    with torch.no_grad():
        weights = model.recurrent_synapse.magnitude.cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=100, color='teal')
    plt.title("SparseConn's Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("weight_distribution_diagnosis.png")
    plt.show()
    
    # 3. CV Distribution
    print("正在计算变异系数 (CV)...")
    cvs = calculate_cv(sample_spikes)
    valid_cvs = cvs[~np.isnan(cvs)]
    if len(valid_cvs) > 0:
        avg_cv = np.mean(valid_cvs)
        print(f"✅ 本次批次的平均CV值为: {avg_cv:.4f}")
        plt.figure(figsize=(10, 6))
        plt.hist(valid_cvs, bins=50, color='purple')
        plt.title("Coefficient of Variation (CV) Distribution")
        plt.xlabel("CV Value")
        plt.ylabel("Neuron Count")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("cv_distribution_diagnosis.png")
        plt.show()
    
    model.train()
    print("--- 可视化诊断结束 ---")


def plot_connectivity_heatmap(conn_matrix, cell_info_df, title='Connectivity Heatmap'):
    """
    根据细胞区域和位置信息对连接矩阵进行排序，并绘制热图。
    本函数遵循 Matrix[Source, Target] 的约定。

    参数:
        conn_matrix (scipy.sparse matrix): 要可视化的连接矩阵。
        cell_info_df (pd.DataFrame): 包含细胞信息的DataFrame，必须有'area'和'bin'列。
        title (str): 图像的标题。
    """
    print(f"\n--- 正在生成'{title}' ---")
    
    # 1. 设置绘图风格
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['lines.markersize'] = 4
    
    # 2. 准备排序所需的数据
    cell_pos = cell_info_df['bin'].to_numpy()
    cell_area_codes, area_labels = pd.factorize(cell_info_df['area'])

    # 3. 计算排序索引 (先按区域area，再按区域内位置pos排序)
    sorted_idx = np.lexsort((cell_pos, cell_area_codes))
    
    # 4. 转换并排序矩阵
    print("正在将稀疏矩阵转换为密集矩阵用于绘图...")
    dense_matrix = conn_matrix.toarray()
    
    print("正在根据脑区和位置重排矩阵...")
    # 根据您的约定 Matrix[Source, Target]，我们对行(Source)和列(Target)进行排序
    sorted_matrix = dense_matrix[sorted_idx, :]
    sorted_matrix = sorted_matrix[:, sorted_idx]
    
    # 5. 绘制热图
    print("正在绘制热图...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_matrix, vmax=np.percentile(sorted_matrix, 90), cmap='viridis', cbar=True)
    
    plt.title(title, fontsize=16)
    
    plt.xlabel("Target Neuron (sorted by region) - down", fontsize=12)
    plt.ylabel("Source Neuron (sorted by region) - up", fontsize=12)
    
    # 添加区域分割线和标签
    unique_areas, counts = np.unique(cell_area_codes, return_counts=True)
    tick_positions = np.cumsum(counts) - counts / 2
    
    plt.xticks(tick_positions, area_labels[unique_areas], rotation=45, ha='right')
    plt.yticks(tick_positions, area_labels[unique_areas], rotation=0)

    for pos in np.cumsum(counts)[:-1]:
        plt.axhline(y=pos, color='white', linewidth=0.5)
        plt.axvline(x=pos, color='white', linewidth=0.5)

    plt.tight_layout()
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(filename)
    print(f"热图已保存为 '{filename}'")
    plt.show()
    
    
'''
def plot_connectivity_heatmap(conn_matrix, cell_info_df, title='Connectivity Heatmap'):
    """根据细胞区域和位置信息对连接矩阵进行排序，并绘制热图。"""
    print(f"\n--- 正在生成'{title}' ---")
    cell_pos = cell_info_df['bin'].to_numpy()
    cell_area_codes, area_labels = pd.factorize(cell_info_df['area'])
    sorted_idx = np.lexsort((cell_pos, cell_area_codes))
    
    dense_matrix = conn_matrix.toarray()
    sorted_matrix = dense_matrix[sorted_idx, :][:, sorted_idx]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_matrix, vmax=np.percentile(sorted_matrix, 90), cmap='viridis', cbar=True)
    plt.title(title, fontsize=16)
    plt.xlabel("Target Neuron (sorted by region)", fontsize=12)
    plt.ylabel("Source Neuron (sorted by region)", fontsize=12)
    
    unique_areas, counts = np.unique(cell_area_codes, return_counts=True)
    tick_positions = np.cumsum(counts) - counts / 2
    plt.xticks(tick_positions, area_labels[unique_areas], rotation=45, ha='right')
    plt.yticks(tick_positions, area_labels[unique_areas], rotation=0)

    for pos in np.cumsum(counts)[:-1]:
        plt.axhline(y=pos, color='white', linewidth=0.5)
        plt.axvline(x=pos, color='white', linewidth=0.5)

    plt.tight_layout()
    filename = title.replace(' ', '_').lower() + '.png'
    plt.savefig(filename)
    plt.show()
'''


# analysis.py

# ... (保留文件顶部的所有 imports 和其他函数) ...

def calculate_branching_ratio(spikes_tensor):
    """
    根据全采样脉冲数据计算网络的分支比 (m)。
    使用 A_{t+1} 对 A_t 的直接线性回归方法。

    参数:
        spikes_tensor (torch.Tensor): 形状为 [Time, Batch, Neurons] 的脉冲张量。

    返回:
        float: 计算出的分支比 m。如果网络活动过低，则返回 np.nan。
    """
    print("\n--- 正在计算网络分支比 (Branching Ratio) ---")
    
    # 确保张量在CPU上并转换为numpy数组
    if isinstance(spikes_tensor, torch.Tensor):
        spikes_tensor = spikes_tensor.cpu().numpy()
        
    # 1. 计算每个时间步的全网络活动 A_t (总脉冲数)
    #    我们使用批次中的第一个样本
    sample_spikes = spikes_tensor[:, 0, :]
    total_activity = np.sum(sample_spikes, axis=1)
    
    # 检查网络是否有足够的活动来进行有意义的回归
    if np.sum(total_activity) < 10 or np.var(total_activity) < 1e-6:
        print("警告: 网络活动过低，无法计算有意义的分支比。")
        return np.nan

    # 2. 创建用于回归的两个序列
    # A_t 是从第0步到倒数第二步的活动
    At = total_activity[:-1]
    # A_{t+1} 是从第1步到最后一步的活动
    At_plus_1 = total_activity[1:]
    
    # (可选优化): 仅在 At > 0 的时间点进行回归，以关注“传播”事件
    # active_indices = np.where(At > 0)[0]
    # if len(active_indices) < 2:
    #     print("警告: 活动传播事件过少，无法计算分支比。")
    #     return np.nan
    # At = At[active_indices]
    # At_plus_1 = At_plus_1[active_indices]

    # 3. 使用 numpy.polyfit 进行一阶线性回归
    # 第三个参数 1 表示进行线性拟合 (y = mx + h)
    # 返回值是 [斜率 m, 截距 h]
    m, h = np.polyfit(At, At_plus_1, 1)
    
    print(f"✅ 计算完成。分支比 (m) ≈ {m:.4f}")
    
    return m