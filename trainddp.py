import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.checkpoint import checkpoint  # 梯度检查点，节省显存
from datetime import timedelta  # 用于DDP超时设置
import matplotlib.pyplot as plt
# 混合精度训练已禁用，使用纯float32训练以保证稳定性
from torch_geometric.loader import DataListLoader # 【关键】使用 ListLoader
import torch.distributed as dist                    # DDP 分布式训练
import torch.multiprocessing as mp                  # 多进程
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP
from torch_geometric.nn import DataParallel         # 兼容性保留
from torch_geometric.data import Batch, Data
import numpy as np
import os
import time
import copy
import json
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 导入自定义模块
# ==========================================
from build_graph import build_graph_data, scan_all_data, root_data_path, load_file_data, is_main_process
from model import BuildGCNList
import os
# DDP环境优化：避免多进程数据加载导致的资源竞争
# 在DDP中，每个GPU进程都会创建num_workers个子进程
# 如果有N个GPU，num_workers=4，则总共有4*N个数据加载子进程
# 这会导致文件描述符耗尽和共享内存对象过多
# 建议：DDP环境中设置num_workers=0，让主进程处理数据加载
NUM_WORKERS = 0  # DDP环境：禁用多进程数据加载以避免资源竞争

# ==========================================
# 2. 训练配置
# ==========================================
# 学习率调度配置
# ==========================================
# ReduceLROnPlateau 配置
REDUCE_LR_PATIENCE = 30  # 多少个epoch无改善就降低学习率
REDUCE_LR_FACTOR = 0.5   # 学习率降低因子
REDUCE_LR_MIN_LR = 1e-6  # 最小学习率
REDUCE_LR_MODE = 'min'   # 'min'表示监控指标越小越好
# REDUCE_LR_VERBOSE 已弃用，使用 get_last_lr() 替代

# ==========================================
# 早停机制配置
# ==========================================
EARLY_STOPPING_PATIENCE = 50  # 连续多少个epoch无改善就停止训练
EARLY_STOPPING_MIN_DELTA = 1e-5  # 最小改善阈值 (0.001e-2 = 1e-5)
EARLY_STOPPING_START_EPOCH = 10  # 从第几个epoch开始检查早停（给模型预热时间）
# DP 模式下 batch_size 是所有卡的总和
# 例如：3张卡，batch_size=48 -> 每张卡分到 16 个图
# 增加 batch size 以提高 GPU 利用率
MATRIX_CACHE = {}
# 优化：DDP模式下每个GPU的batch size
# DDP性能优化：增大batch_size以更好地利用GPU并行计算
# - 24GB 显存：建议 128-256 per GPU
# - 48GB 显存：建议 256-512 per GPU
# - 如果遇到OOM，可以适当减小
TOTAL_BATCH_SIZE = 64  
EPOCH_ADAM = 2000  # 只使用Adam优化器
TOTAL_EPOCHS = EPOCH_ADAM
EPOCH_PRINT = 50
# 从全局配置文件导入参数
from config import N_ITER, SAVE_DIR, OUTPUT_DIR, LOSS_TYPE, MASTER_PORT, LOAD_PRETRAINED_MODEL, PRETRAINED_MODEL_DIR

# ==========================================
# 1.5. 输出目录配置
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 移除模块级别的打印，避免DDP重复打印

LR = 0.001

# 性能优化配置
USE_AMP = False  # 禁用混合精度训练（使用纯float32以保证稳定性）
# torch.compile 配置说明：
# - 需要 PyTorch 2.0+ 支持
# - 可能提升 10-20% 前向传播速度
# - 首次运行需要编译时间（可能较慢）
# - 可能与 DataParallel 和梯度检查点有兼容性问题
# - 建议：先测试是否正常工作，如果遇到错误可以禁用
USE_COMPILE = False # 启用torch.compile可提升10-20%速度（需要PyTorch 2.0+）
PIN_MEMORY = True  # 启用pin_memory以加速数据传输
USE_GRADIENT_CHECKPOINTING = False  # 启用梯度检查点，节省显存（会稍微降低速度，约20-30%）

# ==========================================
# 3. 全局矩阵缓存 (Multi-Device Matrix Cache)
# ==========================================
# 用于解决 DP 模式下不同显卡需要访问不同设备上矩阵的问题

def load_matrix_to_cache(data_mapping, n_total, device_ids, is_main_process=True):
    """
    将所有矩阵预加载到 CPU 上（使用 pin_memory 加速后续传输）。
    结构: MATRIX_CACHE[k_idx] = (A_cpu, b_cpu)

    优化：矩阵存储在 CPU，需要时才转移到 GPU，释放显存空间。
    使用 pin_memory() 加速 CPU 到 GPU 的传输。

    注意: device_ids 参数保留用于兼容性，但不再为每个 GPU 创建副本
    """
    if is_main_process:
        print(f"正在将矩阵预加载到 CPU（使用 pin_memory 优化）...")
        print(f"   矩阵将在需要时动态传输到 GPU，以释放显存空间")
    valid_indices = [k for k in range(1, n_total + 1) if k in data_mapping]
    
    for idx, k_idx in enumerate(valid_indices):
        folder_path, folder_num, data_id = data_mapping[k_idx]
        try:
            # 读取原始数据 (CPU)
            Aij = load_file_data(folder_path, "Aij", folder_num, data_id)
            Av = load_file_data(folder_path, "Av", folder_num, data_id)
            b_data = load_file_data(folder_path, "b", folder_num, data_id)
            
            rows = Aij[:, 0].astype(int) - 1
            cols = Aij[:, 1].astype(int) - 1
            values = Av[:, 0] + 1j * Av[:, 1]
            N_nodes = len(b_data)
            b_val = b_data[:, 0] + 1j * b_data[:, 1]
            
            shape = (N_nodes, N_nodes)
            
            # 在 CPU 上创建张量，并使用 pin_memory 加速后续传输
            i = torch.from_numpy(np.vstack((rows, cols))).long()
            # v = torch.from_numpy(values.astype(np.complex128))  # 使用双精度complex
            # b_k = torch.from_numpy(b_val.astype(np.complex128))  # 使用双精度complex
            # 改为：
            v = torch.from_numpy(values.astype(np.complex64))  # 32位复数
            b_k = torch.from_numpy(b_val.astype(np.complex64))

            # # 将A矩阵和b向量都放大10倍
            # v = v * 10
            # b_k = b_k * 10

            # 使用 pin_memory() 将数据固定在内存中，加速 CPU->GPU 传输
            # 注意：稀疏矩阵的 indices 和 values 可以 pin_memory
            i = i.pin_memory()
            v = v.pin_memory()
            b_k = b_k.pin_memory()
            
            # 在 CPU 上创建稀疏矩阵（不 coalesce，延迟到 GPU 传输时）
            # 存储 indices 和 values（已 pin_memory）以及 shape，而不是完整的稀疏矩阵
            # 这样可以保持 pin_memory 状态
            MATRIX_CACHE[k_idx] = {
                'indices': i,
                'values': v,
                'shape': shape,
                'b': b_k
            }
                
            if is_main_process and (idx + 1) % 1000 == 0:
                print(f"  已缓存 {idx + 1}/{len(valid_indices)} 个样本")

        except Exception as e:
            if is_main_process:
                print(f"加载样本 {k_idx} 出错: {e}")
            continue
    if is_main_process:
        print("矩阵缓存完成（存储在 CPU，使用 pin_memory 优化，双精度complex128）。")
        # 验证精度设置
        if MATRIX_CACHE:
            sample_key = list(MATRIX_CACHE.keys())[0]
            sample_data = MATRIX_CACHE[sample_key]
            print(f"示例矩阵精度检查: values.dtype={sample_data['values'].dtype}, b.dtype={sample_data['b'].dtype}")

def get_Ab(k_idx, device, dtype=None):
    """
    从缓存中获取当前设备对应的 A 和 b。
    如果矩阵在 CPU 上，则动态传输到目标设备（使用 non_blocking 加速）。

    Args:
        k_idx: 样本索引
        device: 目标设备
        dtype: 目标数据类型，如果为None则保持原有精度

    优化：
    1. 矩阵存储在 CPU，需要时才传输到 GPU
    2. 使用 non_blocking=True 进行异步传输
    3. 使用 pin_memory 加速传输
    """
    if k_idx not in MATRIX_CACHE:
        raise RuntimeError(f"未找到样本 k={k_idx} 的缓存数据")
    
    cache_data = MATRIX_CACHE[k_idx]
    
    # 如果目标设备是 CPU，在 CPU 上构建稀疏矩阵
    if device.type == 'cpu':
        indices = cache_data['indices']
        values = cache_data['values']
        shape = cache_data['shape']
        b_cpu = cache_data['b']

        # 根据需要转换数据类型
        if dtype is not None:
            values = values.to(dtype)
            b_cpu = b_cpu.to(dtype)

        A_cpu = torch.sparse_coo_tensor(indices, values, shape, device=torch.device('cpu'))
        return A_cpu, b_cpu
    
    # 如果目标设备是 GPU，将矩阵传输到 GPU
    # 使用 non_blocking=True 进行异步传输（需要 pin_memory）
    try:
        # 获取已 pin_memory 的 indices 和 values
        indices_cpu = cache_data['indices']
        values_cpu = cache_data['values']
        shape = cache_data['shape']
        b_cpu = cache_data['b']

        # 根据需要转换数据类型
        if dtype is not None:
            values_cpu = values_cpu.to(dtype)
            b_cpu = b_cpu.to(dtype)

        # 异步传输到 GPU（non_blocking 需要 pin_memory）
        indices_gpu = indices_cpu.to(device, non_blocking=True)
        values_gpu = values_cpu.to(device, non_blocking=True)

        # 在 GPU 上重建稀疏矩阵并 coalesce
        A_gpu = torch.sparse_coo_tensor(indices_gpu, values_gpu, shape, device=device).coalesce()

        # 异步传输 b 向量
        b_gpu = b_cpu.to(device, non_blocking=True)

        return A_gpu, b_gpu
    except Exception as e:
        # 如果异步传输失败，回退到同步传输
        if is_main_process():
            print(f"警告: 异步传输失败，使用同步传输 (k={k_idx}): {e}")
        indices_cpu = cache_data['indices']
        values_cpu = cache_data['values']
        shape = cache_data['shape']
        b_cpu = cache_data['b']

        # 根据需要转换数据类型
        if dtype is not None:
            values_cpu = values_cpu.to(dtype)
            b_cpu = b_cpu.to(dtype)

        indices_gpu = indices_cpu.to(device)
        values_gpu = values_cpu.to(device)
        A_gpu = torch.sparse_coo_tensor(indices_gpu, values_gpu, shape, device=device).coalesce()
        b_gpu = b_cpu.to(device)

        return A_gpu, b_gpu

# ==========================================
# 3. 辅助函数：绘制训练曲线
# ==========================================
def plot_training_curve(train_losses=None, test_losses=None, save_path="training_curve.svg", data_file=None):
    """
    绘制训练集和测试集loss变化曲线并保存为文件

    Args:
        train_losses: 训练集loss列表（可选，如果提供data_file则忽略）
        test_losses: 测试集loss列表（可选，如果提供data_file则忽略）
        save_path: 保存路径
        data_file: 训练数据JSON文件路径，如果提供则从文件加载数据
    """
    # 在DDP环境中，只在主进程中执行绘图
    if not is_main_process():
        return
    if data_file is not None:
        # 从文件加载数据
        data = load_training_data(data_file)
        if data is None:
            if is_main_process():
                print("❌ 无法加载训练数据，跳过绘图")
            return
        train_losses = data.get('train_losses', [])
        test_losses = data.get('test_losses', [])
        if not train_losses or not test_losses:
            if is_main_process():
                print("❌ 训练数据中缺少loss信息，跳过绘图")
            return
    plt.figure(figsize=(12, 6))

    epochs = range(1, len(train_losses) + 1)

    # 绘制训练loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数尺度

    # 绘制测试loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数尺度

    # 保存为矢量图格式
    # PDF格式（高质量打印）
    pdf_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).replace('.png', '.pdf'))
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    if is_main_process():
        print(f"✅ 训练曲线PDF已保存到: {pdf_path}")

    # SVG格式（网页和现代应用）
    plt.figure(figsize=(12, 6))

    # 重新绘制（合并在一个图中）
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数尺度

    svg_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).replace('.png', '.svg'))
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()

    if is_main_process():
        print(f"✅ 训练曲线SVG已保存到: {svg_path}")


def plot_mse_res_loss(train_mse=None, test_mse=None, train_res=None, test_res=None, save_path="mse_res_loss.svg", data_file=None):
    """
    绘制包含MSE loss和RES loss的训练曲线

    Args:
        train_mse: 训练集MSE loss列表（可选，如果提供data_file则忽略）
        test_mse: 测试集MSE loss列表（可选，如果提供data_file则忽略）
        train_res: 训练集RES loss列表（可选，如果提供data_file则忽略）
        test_res: 测试集RES loss列表（可选，如果提供data_file则忽略）
        save_path: 保存路径
        data_file: 训练数据JSON文件路径，如果提供则从文件加载数据
    """
    # 在DDP环境中，只在主进程中执行绘图
    if not is_main_process():
        return
    if data_file is not None:
        # 从文件加载数据
        data = load_training_data(data_file)
        if data is None:
            if is_main_process():
                print("❌ 无法加载训练数据，跳过绘图")
            return
        train_mse = data.get('train_mse_losses', [])
        test_mse = data.get('test_mse_losses', [])
        train_res = data.get('train_res_losses', [])
        test_res = data.get('test_res_losses', [])
        if not all([train_mse, test_mse, train_res, test_res]):
            if is_main_process():
                print("❌ 训练数据中缺少MSE或RES loss信息，跳过绘图")
            return
    plt.figure(figsize=(14, 6))
    
    epochs = range(1, len(train_mse) + 1)
    
    # 绘制MSE loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_mse, 'b-', label='Training MSE', linewidth=2, marker='o', markersize=3)
    plt.plot(epochs, test_mse, 'r-', label='Testing MSE', linewidth=2, marker='+', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('MSE Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # 绘制RES loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_res, 'b-', label='Training Res', linewidth=2, marker='o', markersize=3)
    plt.plot(epochs, test_res, 'r-', label='Testing Res', linewidth=2, marker='+', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('RES loss')
    plt.title('Residual Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.2)  # 设置y轴范围为0~0.2
    
    # 保存为矢量图格式 (SVG)
    svg_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).replace('.png', '.svg'))
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()

    if is_main_process():
        print(f"✅ MSE和RES loss曲线SVG已保存到: {svg_path}")


def plot_mse_loss_distribution(train_mse_losses=None, test_mse_losses=None, save_path="mse_loss_distribution.svg", data_file=None):
    """
    绘制MSE loss分布柱状图

    Args:
        train_mse_losses: 训练集MSE loss列表（可选，如果提供data_file则忽略）
        test_mse_losses: 测试集MSE loss列表（可选，如果提供data_file则忽略）
        save_path: 保存路径
        data_file: 训练数据JSON文件路径，如果提供则从文件加载数据
    """
    # 在DDP环境中，只在主进程中执行绘图
    if not is_main_process():
        return
    if data_file is not None:
        # 从文件加载数据
        data = load_training_data(data_file)
        if data is None:
            if is_main_process():
                print("❌ 无法加载训练数据，跳过绘图")
            return
        train_mse_losses = data.get('train_mse_losses', [])
        test_mse_losses = data.get('test_mse_losses', [])
        if not train_mse_losses or not test_mse_losses:
            if is_main_process():
                print("❌ 训练数据中缺少MSE loss信息，跳过绘图")
            return
    plt.figure(figsize=(12, 6))

    # 确保输入是有效的数值列表
    if not train_mse_losses or not test_mse_losses:
        if is_main_process():
            print("⚠️  警告：训练集或测试集MSE loss数据为空，跳过绘图")
        plt.close()
        return

    # 转换为numpy数组并过滤无效值
    train_mse_losses = np.array(train_mse_losses, dtype=np.float32)
    test_mse_losses = np.array(test_mse_losses, dtype=np.float32)

    # 过滤掉NaN和inf值
    train_mse_losses = train_mse_losses[np.isfinite(train_mse_losses)]
    test_mse_losses = test_mse_losses[np.isfinite(test_mse_losses)]

    if len(train_mse_losses) == 0 or len(test_mse_losses) == 0:
        if is_main_process():
            print("⚠️  警告：过滤后训练集或测试集MSE loss数据为空，跳过绘图")
        plt.close()
        return

    # 计算统计信息
    train_mean = np.mean(train_mse_losses)
    train_std = np.std(train_mse_losses)
    test_mean = np.mean(test_mse_losses)
    test_std = np.std(test_mse_losses)

    # 设置bins（对数坐标需要特殊的处理）
    all_mse_losses = np.concatenate([train_mse_losses, test_mse_losses])
    # 确保所有值都是正数（MSE loss应该都是正数）
    all_mse_losses = all_mse_losses[all_mse_losses > 0]
    if len(all_mse_losses) == 0:
        if is_main_process():
            print("⚠️  警告：所有MSE loss值都是非正数，跳过绘图")
        plt.close()
        return

    # 创建对数bins
    log_min = np.log10(max(all_mse_losses.min(), 1e-10))  # 避免log(0)
    log_max = np.log10(all_mse_losses.max())
    bins = np.logspace(log_min, log_max, 50)

    # 计算直方图数据
    train_hist, _ = np.histogram(train_mse_losses, bins=bins, density=True)
    test_hist, _ = np.histogram(test_mse_losses, bins=bins, density=True)

    # 计算bin中心用于绘制
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 设置bar宽度
    bar_width = np.diff(bins) * 0.8  # 每个bin的80%宽度

    # 绘制训练集和测试集的柱状图（分开放置）
    plt.bar(bin_centers - bar_width/4, train_hist, width=bar_width/2, alpha=0.8, color='coral',
            label=f'Training (μ={train_mean:.3f}, σ={train_std:.5f})', edgecolor='black', linewidth=0.5)

    plt.bar(bin_centers + bar_width/4, test_hist, width=bar_width/2, alpha=0.8, color='cyan',
            label=f'Testing (μ={test_mean:.3f}, σ={test_std:.5f})', edgecolor='black', linewidth=0.5)

    # 拟合高斯分布并绘制
    from scipy import stats

    # 训练集高斯拟合
    train_params = stats.norm.fit(train_mse_losses)
    train_x = np.logspace(log_min, log_max, 100)  # 在对数空间均匀分布的点用于显示
    train_pdf = stats.norm.pdf(train_x, *train_params)
    plt.plot(train_x, train_pdf, 'r-', linewidth=2, label='Training Gaussian Fit')

    # 测试集高斯拟合
    test_params = stats.norm.fit(test_mse_losses)
    test_x = np.logspace(log_min, log_max, 100)  # 在对数空间均匀分布的点用于显示
    test_pdf = stats.norm.pdf(test_x, *test_params)
    plt.plot(test_x, test_pdf, 'b-', linewidth=2, label='Testing Gaussian Fit')

    plt.xlabel('MSE Loss (log scale)')
    plt.ylabel('Density')
    plt.title('MSE Loss Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')  # 设置x轴为对数坐标
    
    # 保存为矢量图格式 (SVG)
    svg_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).replace('.png', '.svg'))
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()

    if is_main_process():
        print(f"✅ MSE loss分布图SVG已保存到: {svg_path}")
        print(f"   训练集: 均值={train_mean:.6f}, 标准差={train_std:.6f}")
        print(f"   测试集: 均值={test_mean:.6f}, 标准差={test_std:.6f}")


def compute_relative_errors(solver, data_loader, data_mapping, device, matrix_dtype):
    """
    计算数据集中每个样本的相对误差
    
    Args:
        solver: 训练好的模型
        data_loader: 数据加载器
        data_mapping: 数据映射字典
        device: 设备
        matrix_dtype: 矩阵数据类型
    
    Returns:
        relative_errors: 每个样本的相对误差列表
    """
    solver.eval()
    relative_errors = []
    
    with torch.no_grad():
        for data_list in data_loader:
            # 获取batch
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            processed_data_list = []
            for item in data_list:
                if isinstance(item, Data):
                    if not hasattr(item, 'k_idx'):
                        item.k_idx = torch.tensor([0])
                    elif not isinstance(item.k_idx, torch.Tensor):
                        item.k_idx = torch.tensor([item.k_idx] if not isinstance(item.k_idx, (list, tuple)) else item.k_idx)
                    processed_data_list.append(item)
                elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], Data):
                    processed_data_list.append(item[0])
            
            try:
                batch = Batch.from_data_list(processed_data_list)
            except:
                from torch_geometric.data.collate import collate
                batch, _, _ = collate(Data, processed_data_list, increment=True, add_batch=True, follow_batch=[])
            
            batch = batch.to(device)
            k_all = batch.k_idx
            node_batch = batch.batch
            B = k_all.size(0)
            
            # 获取模型精度 - 兼容不同层类型
            raw_model = solver.module if isinstance(solver, (DataParallel, DDP)) else solver
            if hasattr(raw_model.model_real.networks[0].gcn1, 'conv'):
                is_double_precision = raw_model.model_real.networks[0].gcn1.conv.lin_fusion.weight.dtype == torch.float64
            elif hasattr(raw_model.model_real.networks[0].gcn1, 'spatial_conv'):
                is_double_precision = raw_model.model_real.networks[0].gcn1.spatial_conv.lin_fusion.weight.dtype == torch.float64
            elif hasattr(raw_model.model_real.networks[0].gcn1, 'linear'):
                is_double_precision = raw_model.model_real.networks[0].gcn1.linear.weight.dtype == torch.float64
            else:
                is_double_precision = False

            if is_double_precision:
                data_dtype = torch.float64
            else:
                data_dtype = torch.float32
            
            # 准备数据
            eps_feat = batch.x[:, 0:2].to(data_dtype)
            current_E_real = batch.x[:, 4].to(data_dtype)
            current_E_imag = batch.x[:, 5].to(data_dtype)
            true_real = batch.y[:, 0].to(data_dtype)
            true_imag = batch.y[:, 1].to(data_dtype)
            
            # 加载矩阵
            A_list = []
            b_list = []
            for b_idx in range(B):
                k = int(k_all[b_idx].item())
                A, b = get_Ab(k, device, matrix_dtype)
                A_list.append(A)
                b_list.append(b)
            
            # 前向传播
            E_real_cur = current_E_real
            E_imag_cur = current_E_imag
            
            for iter_idx in range(raw_model.n_iter):
                # 计算残差
                r_real_list = []
                r_imag_list = []
                for b_idx in range(B):
                    mask = (node_batch == b_idx)
                    E_r = E_real_cur[mask]
                    E_i = E_imag_cur[mask]
                    A = A_list[b_idx]
                    b_vec = b_list[b_idx]
                    E_c = torch.complex(E_r, E_i)
                    Ax = torch.sparse.mm(A, E_c.unsqueeze(-1)).squeeze(-1)
                    r_c = b_vec - Ax
                    if data_dtype == torch.float64:
                        r_real_list.append(r_c.real.double())
                        r_imag_list.append(r_c.imag.double())
                    else:
                        r_real_list.append(r_c.real.float())
                        r_imag_list.append(r_c.imag.float())
                    del E_c, Ax, r_c
                
                r_real = torch.cat(r_real_list, dim=0)
                r_imag = torch.cat(r_imag_list, dim=0)
                del r_real_list, r_imag_list
                
                # GNN前向
                # 从batch.x中提取背景场
                bg_real = batch.x[:, 6].to(data_dtype)
                bg_imag = batch.x[:, 7].to(data_dtype)
                x_in = torch.cat([
                    eps_feat,
                    r_real.view(-1, 1),
                    r_imag.view(-1, 1),
                    E_real_cur.view(-1, 1),  # 当前电场实部 (随迭代更新)
                    E_imag_cur.view(-1, 1),  # 当前电场虚部 (随迭代更新)
                    bg_real.view(-1, 1),     # 背景场实部 (不随网络更新)
                    bg_imag.view(-1, 1)      # 背景场虚部 (不随网络更新)
                ], dim=1)

                delta_real = raw_model.model_real(x_in, batch.edge_index, batch.batch, iter_idx)
                delta_imag = raw_model.model_imag(x_in, batch.edge_index, batch.batch, iter_idx)
                
                E_real_cur = E_real_cur + delta_real.view(-1)
                E_imag_cur = E_imag_cur + delta_imag.view(-1)
                del x_in, delta_real, delta_imag, r_real, r_imag
            
            # 计算每个样本的相对误差
            for b_idx in range(B):
                mask = (node_batch == b_idx)
                pred_real = E_real_cur[mask]
                pred_imag = E_imag_cur[mask]
                true_r = true_real[mask]
                true_i = true_imag[mask]
                
                # 计算平方的相对误差: ||pred - true||_2^2 / ||true||_2^2（与公式一致）
                # 分子：||pred - true||_2^2
                numerator = ((pred_real - true_r).pow(2).sum() + (pred_imag - true_i).pow(2).sum())
                # 分母：||true||_2^2
                denominator = (true_r.pow(2).sum() + true_i.pow(2).sum())
                
                if denominator > 1e-10:  # 避免除零
                    # 平方的相对误差：||x^FEM - x^GCN||_2^2 / ||x^FEM||_2^2
                    rel_error_squared = (numerator / denominator).item()
                    relative_errors.append(rel_error_squared)
            
            del A_list, b_list, E_real_cur, E_imag_cur
    
    return relative_errors

def save_training_data(training_data, save_path="training_data.json"):
    """
    保存训练数据到JSON文件

    Args:
        training_data: 包含训练数据的字典
        save_path: 保存路径
    """
    # 在DDP环境中，只在主进程中保存数据
    if not is_main_process():
        return
    import json

    # 将numpy类型转换为Python原生类型，以便JSON序列化
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    serializable_data = convert_to_serializable(training_data)

    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)

    if is_main_process():
        print(f"✅ 训练数据已保存到: {save_path}")

def load_training_data(load_path="training_data.json"):
    """
    从JSON文件加载训练数据

    Args:
        load_path: 加载路径

    Returns:
        dict: 训练数据字典
    """
    import json

    try:
        with open(load_path, 'r') as f:
            data = json.load(f)
        if is_main_process():
            print(f"✅ 训练数据已从 {load_path} 加载")
        return data
    except FileNotFoundError:
        if is_main_process():
            print(f"⚠️  警告：找不到训练数据文件 {load_path}")
        return None
    except Exception as e:
        if is_main_process():
            print(f"❌ 加载训练数据失败: {e}")
        return None

# ==========================================
# 3. 辅助函数：处理 DataParallel 的输出
# ==========================================
def extract_loss_and_num_nodes(outputs):
    """
    从 DataParallel 的输出中提取 loss 和 num_nodes。

    注意：forward 方法返回单个 tensor：
    - 普通模式：[loss_sum, res_sum, num_nodes, num_samples] (4个值)
    - Hybrid模式：[loss_sum, res_sum, num_nodes, num_samples, mse_loss_sum, phi_loss_sum] (6个值)

    Args:
        outputs: DataParallel 返回的结果
            - 如果是列表，每个元素是上述格式的 tensor
            - 如果是单个 tensor，形状为 [4]/[6] 或 [N, 4]/[N, 6]（N 个 GPU）

    Returns:
        tuple: (total_loss_sum, total_res_sum, total_num_nodes, total_num_samples, total_mse_sum, total_phi_sum)
            - total_loss_sum: 所有 GPU 的 loss 总和 (tensor)
            - total_res_sum: 所有 GPU 的 RES loss 总和 (tensor)
            - total_num_nodes: 所有 GPU 的 num_nodes 总和 (int)
            - total_num_samples: 所有 GPU 的 num_samples 总和 (int)
            - total_mse_sum: 所有 GPU 的 MSE loss 总和 (tensor，仅在hybrid模式下有效)
            - total_phi_sum: 所有 GPU 的 Phi loss 总和 (tensor，仅在hybrid模式下有效)
    """
    # 检查 outputs 是否为空（只检查 None 和空列表，不检查张量）
    if outputs is None:
        return torch.tensor(0.0), torch.tensor(0.0), 0, 0, torch.tensor(0.0), torch.tensor(0.0)
    if isinstance(outputs, list) and len(outputs) == 0:
        return torch.tensor(0.0), torch.tensor(0.0), 0, 0, torch.tensor(0.0), torch.tensor(0.0)

    # 处理 DataParallel 的输出格式
    # DataParallel 可能返回：
    # 1. 列表: [[loss1, res1, n1, s1, mse1, phi1], [loss2, res2, n2, s2, mse2, phi2], ...] (hybrid模式)
    # 2. 列表: [[loss1, res1, n1, s1], [loss2, res2, n2, s2], ...] (普通模式)
    # 3. 单个 tensor: [[loss1, res1, n1, s1, mse1, phi1], ...] 形状为 [N, 6] 或 [N, 4]
    # 4. 单个 tensor: [loss, res, n, s, mse, phi] 形状为 [6] 或 [4]（单 GPU 情况）

    if isinstance(outputs, list):
        # 列表格式：每个元素是 [loss, res, num_nodes, num_samples, mse?, phi?] 的 tensor
        loss_list = []
        res_list = []
        num_nodes_list = []
        num_samples_list = []
        mse_list = []
        phi_list = []
        is_hybrid_mode = False

        for o in outputs:
            if isinstance(o, torch.Tensor):
                if o.dim() == 1:
                    if o.shape[0] == 4:
                        # 普通模式：形状为 [4] 的 tensor
                        loss_list.append(o[0])
                        res_list.append(o[1])
                        num_nodes_list.append(o[2].item())
                        num_samples_list.append(o[3].item())
                        mse_list.append(torch.tensor(0.0))  # 占位符
                        phi_list.append(torch.tensor(0.0))  # 占位符
                    elif o.shape[0] == 6:
                        # Hybrid模式：形状为 [6] 的 tensor
                        is_hybrid_mode = True
                        loss_list.append(o[0])
                        res_list.append(o[1])
                        num_nodes_list.append(o[2].item())
                        num_samples_list.append(o[3].item())
                        mse_list.append(o[4])
                        phi_list.append(o[5])
                    else:
                        raise ValueError(f"不支持的 tensor 形状: {o.shape}")
                elif o.dim() == 2:
                    if o.shape[1] == 4:
                        # 普通模式：形状为 [N, 4] 的 tensor（多个 GPU 合并）
                        loss_list.append(o[:, 0].sum())
                        res_list.append(o[:, 1].sum())
                        num_nodes_list.append(o[:, 2].sum().item())
                        num_samples_list.append(o[:, 3].sum().item())
                        mse_list.append(torch.tensor(0.0))  # 占位符
                        phi_list.append(torch.tensor(0.0))  # 占位符
                    elif o.shape[1] == 6:
                        # Hybrid模式：形状为 [N, 6] 的 tensor（多个 GPU 合并）
                        is_hybrid_mode = True
                        loss_list.append(o[:, 0].sum())
                        res_list.append(o[:, 1].sum())
                        num_nodes_list.append(o[:, 2].sum().item())
                        num_samples_list.append(o[:, 3].sum().item())
                        mse_list.append(o[:, 4].sum())
                        phi_list.append(o[:, 5].sum())
                    else:
                        raise ValueError(f"不支持的 tensor 形状: {o.shape}")
                else:
                    raise ValueError(f"不支持的 tensor 形状: {o.shape}")
            else:
                raise TypeError(f"不支持的输出类型: {type(o)}")

        total_loss = torch.stack(loss_list).sum() if loss_list else torch.tensor(0.0)
        total_res = torch.stack(res_list).sum() if res_list else torch.tensor(0.0)
        total_num_nodes = sum(num_nodes_list)
        total_num_samples = sum(num_samples_list)
        total_mse = torch.stack(mse_list).sum() if is_hybrid_mode and mse_list else torch.tensor(0.0)
        total_phi = torch.stack(phi_list).sum() if is_hybrid_mode and phi_list else torch.tensor(0.0)
        
    elif isinstance(outputs, torch.Tensor):
        # 单个 tensor 格式
        if outputs.dim() == 1:
            if outputs.shape[0] == 4:
                # 普通模式：形状为 [4]：单 GPU
                total_loss = outputs[0]
                total_res = outputs[1]
                total_num_nodes = int(outputs[2].item())
                total_num_samples = int(outputs[3].item())
                total_mse = torch.tensor(0.0)
                total_phi = torch.tensor(0.0)
            elif outputs.shape[0] == 6:
                # Hybrid模式：形状为 [6]：单 GPU
                total_loss = outputs[0]
                total_res = outputs[1]
                total_num_nodes = int(outputs[2].item())
                total_num_samples = int(outputs[3].item())
                total_mse = outputs[4]
                total_phi = outputs[5]
            elif outputs.shape[0] % 4 == 0 and outputs.shape[0] % 6 != 0:
                # 普通模式：形状为 [4*N]：多个 GPU 的输出被展平（例如 [12] = 3个GPU * 4）
                # 需要重塑为 [N, 4] 格式
                n_gpus = outputs.shape[0] // 4
                outputs_reshaped = outputs.view(n_gpus, 4)
                total_loss = outputs_reshaped[:, 0].sum()
                total_res = outputs_reshaped[:, 1].sum()
                total_num_nodes = int(outputs_reshaped[:, 2].sum().item())
                total_num_samples = int(outputs_reshaped[:, 3].sum().item())
                total_mse = torch.tensor(0.0)
                total_phi = torch.tensor(0.0)
            elif outputs.shape[0] % 6 == 0:
                # Hybrid模式：形状为 [6*N]：多个 GPU 的输出被展平
                # 需要重塑为 [N, 6] 格式
                n_gpus = outputs.shape[0] // 6
                outputs_reshaped = outputs.view(n_gpus, 6)
                total_loss = outputs_reshaped[:, 0].sum()
                total_res = outputs_reshaped[:, 1].sum()
                total_num_nodes = int(outputs_reshaped[:, 2].sum().item())
                total_num_samples = int(outputs_reshaped[:, 3].sum().item())
                total_mse = outputs_reshaped[:, 4].sum()
                total_phi = outputs_reshaped[:, 5].sum()
            else:
                raise ValueError(f"不支持的 tensor 形状: {outputs.shape}（一维张量长度必须是4或6的倍数）")
        elif outputs.dim() == 2:
            if outputs.shape[1] == 4:
                # 普通模式：形状为 [N, 4]：多个 GPU 合并
                total_loss = outputs[:, 0].sum()
                total_res = outputs[:, 1].sum()
                total_num_nodes = int(outputs[:, 2].sum().item())
                total_num_samples = int(outputs[:, 3].sum().item())
                total_mse = torch.tensor(0.0)
                total_phi = torch.tensor(0.0)
            elif outputs.shape[1] == 6:
                # Hybrid模式：形状为 [N, 6]：多个 GPU 合并
                total_loss = outputs[:, 0].sum()
                total_res = outputs[:, 1].sum()
                total_num_nodes = int(outputs[:, 2].sum().item())
                total_num_samples = int(outputs[:, 3].sum().item())
                total_mse = outputs[:, 4].sum()
                total_phi = outputs[:, 5].sum()
            else:
                raise ValueError(f"不支持的 tensor 形状: {outputs.shape}")
        else:
            raise ValueError(f"不支持的 tensor 形状: {outputs.shape}")
    else:
        raise TypeError(f"不支持的输出类型: {type(outputs)}")

    return total_loss, total_res, total_num_nodes, total_num_samples, total_mse, total_phi


# ==========================================
# 4. Loss计算函数
# ==========================================
def compute_mse_loss(E_real_cur, E_imag_cur, batch_y, num_nodes=None):
    """
    计算传统的MSE损失：||pred - true||^2（返回总和，未平均）

    Args:
        E_real_cur: 预测的实部电场 [N]
        E_imag_cur: 预测的虚部电场 [N]
        batch_y: 真实标签 [N, 2]
        num_nodes: 节点总数（用于接口统一，不使用）

    Returns:
        loss_sum: MSE损失总和（未平均）
    """
    true_real = batch_y[:, 0]
    true_imag = batch_y[:, 1]

    # 计算实部和虚部的平方误差总和
    term1 = (E_real_cur - true_real).pow(2).sum()  # 实部误差平方和
    term2 = (E_imag_cur - true_imag).pow(2).sum()  # 虚部误差平方和

    # 返回总SSE（Sum of Squared Errors）
    loss_sum = 0.5 * (term1 + term2)

    # 释放中间变量
    del term1, term2

    return loss_sum

def compute_phi_loss(E_real_cur, E_imag_cur, k_all, node_batch, B, device, matrix_dtype, num_nodes=None):
    """
    计算Phi损失：||A*x - b||^2（物理残差，不需要节点平均）

    Args:
        E_real_cur: 预测的实部电场 [N]
        E_imag_cur: 预测的虚部电场 [N]
        k_all: 样本索引 [B]
        node_batch: 节点到批次的映射 [N]
        B: 批次大小
        device: 计算设备
        matrix_dtype: 矩阵数据类型
        num_nodes: 节点总数（Phi loss不使用，用于接口统一）

    Returns:
        loss_sum: Phi损失总和（不进行节点平均）
    """
    phi_losses = []
    for b_idx in range(B):
        k = int(k_all[b_idx].item())
        A, b = get_Ab(k, device, matrix_dtype)

        mask = (node_batch == b_idx)
        E_r = E_real_cur[mask]
        E_i = E_imag_cur[mask]

        E_c = torch.complex(E_r, E_i)
        Ax = torch.sparse.mm(A, E_c.unsqueeze(-1)).squeeze(-1)
        r = b - Ax

        # 计算残差的L2范数平方: ||Ax - b||_2^2
        phi_loss = torch.norm(r, p=2).pow(2)
        phi_losses.append(phi_loss)

    # 使用torch.stack和torch.sum保持梯度计算图
    loss_sum = torch.stack(phi_losses).sum()

    return loss_sum

def compute_asinh_loss(E_real_cur, E_imag_cur, batch_y, num_nodes):
    """
    计算Asinh损失：sqrt(asinh(||pred - true||^2))（不需要节点平均）

    Args:
        E_real_cur: 预测的实部电场 [N]
        E_imag_cur: 预测的虚部电场 [N]
        batch_y: 真实标签 [N, 2]
        num_nodes: 节点总数（Asinh loss不使用，用于接口统一）

    Returns:
        loss_value: Asinh损失（不进行节点平均）
    """
    true_real = batch_y[:, 0]
    true_imag = batch_y[:, 1]

    # 计算预测值与真实值的差
    diff_real = E_real_cur - true_real
    diff_imag = E_imag_cur - true_imag

    # 分别计算实部和虚部的平均平方误差
    mse_real = diff_real.pow(2).mean()  # 实部平均平方误差
    mse_imag = diff_imag.pow(2).mean()  # 虚部平均平方误差

    # 分别应用asinh函数
    asinh_real = torch.asinh(mse_real)
    asinh_imag = torch.asinh(mse_imag)

    # 将实部和虚部的asinh结果相加，然后开方
    loss_sum = torch.sqrt(asinh_real + asinh_imag)

    return loss_sum

def compute_hybrid_loss(E_real_cur, E_imag_cur, batch_y, k_all, node_batch, B, device, matrix_dtype, epoch, num_nodes):
    """
    计算Hybrid损失：始终采用MSE + 0.1*Phi的固定组合（Phi loss进行节点平均）

    Args:
        E_real_cur: 预测的实部电场 [N]
        E_imag_cur: 预测的虚部电场 [N]
        batch_y: 真实标签 [N, 2]
        k_all: 样本索引 [B]
        node_batch: 节点到批次的映射 [N]
        B: 批次大小
        device: 计算设备
        matrix_dtype: 矩阵数据类型
        epoch: 当前训练轮次
        num_nodes: 节点总数

    Returns:
        tuple: (loss_value, mse_loss_sum, phi_loss_sum)
        - loss_value: Hybrid损失（MSE + 0.1*Phi，进行Phi节点平均）
        - mse_loss_sum: MSE loss总和（未平均）
        - phi_loss_sum: Phi loss总和（未平均）
    """
    # 计算MSE loss（总和，未平均）
    mse_loss_sum = compute_mse_loss(E_real_cur, E_imag_cur, batch_y, num_nodes)

    # 计算Phi loss（总和，未平均）
    phi_loss_sum = compute_phi_loss(E_real_cur, E_imag_cur, k_all, node_batch, B, device, matrix_dtype, num_nodes)

    # Phi loss进行节点平均：总Phi loss除以节点数num_nodes
    phi_loss_per_node = phi_loss_sum / num_nodes

    # 固定权重策略：始终使用MSE + 0.1*Phi
    lambda_phi = 0.5

    # 组合损失：(MSE_sum / num_nodes) + 0.1 * Phi_per_node
    loss_value = (mse_loss_sum / num_nodes) + lambda_phi * phi_loss_per_node

    return loss_value, mse_loss_sum, phi_loss_sum

# ==========================================
# ==========================================
# 3.5. 预训练模型加载函数
# ==========================================
def load_pretrained_model(model_dir, solver, n_iter, device, is_main_process):
    """
    只加载预训练模型权重（不加载优化器、调度器等状态）
    用于迁移学习：在新数据集上使用预训练模型继续训练
    
    Args:
        model_dir: 模型权重文件目录
        solver: 模型
        n_iter: 迭代次数
        device: 设备
        is_main_process: 是否为主进程
    
    Returns:
        bool: 是否成功加载
    """
    if is_main_process:
        print(f"📂 正在加载预训练模型权重: {model_dir}")
    
    # 获取原始模型（去除DDP包装）
    raw_model = solver.module if isinstance(solver, (DataParallel, DDP)) else solver
    
    # 检查模型文件是否存在
    all_exist = True
    for i in range(n_iter):
        real_path = os.path.join(model_dir, f"real_iter_{i}.pth")
        imag_path = os.path.join(model_dir, f"imag_iter_{i}.pth")
        if not os.path.exists(real_path) or not os.path.exists(imag_path):
            all_exist = False
            if is_main_process:
                print(f"⚠️  模型文件不存在: {real_path} 或 {imag_path}")
            break
    
    if not all_exist:
        if is_main_process:
            print("   将从随机初始化开始训练")
        return False
    
    # 加载模型权重
    try:
        for i in range(n_iter):
            real_path = os.path.join(model_dir, f"real_iter_{i}.pth")
            imag_path = os.path.join(model_dir, f"imag_iter_{i}.pth")
            
            real_net = raw_model.model_real.get_network(i)
            imag_net = raw_model.model_imag.get_network(i)
            
            real_net.load_state_dict(torch.load(real_path, map_location=device, weights_only=True))
            imag_net.load_state_dict(torch.load(imag_path, map_location=device, weights_only=True))
        
        if is_main_process:
            print(f"✅ 预训练模型权重加载成功!")
            print(f"   已加载 {n_iter} 个迭代网络的权重")
            print(f"   ⚠️  注意：优化器、学习率调度器等状态已重新初始化")
        
        return True
    except Exception as e:
        if is_main_process:
            print(f"❌ 加载预训练模型失败: {e}")
            print("   将从随机初始化开始训练")
        return False


# 4. 物理求解器封装 (核心逻辑)
# ==========================================
class PhiSAGESolver(nn.Module):
    """
    将物理迭代循环封装为 Module，以便 DataParallel 可以自动分发计算。
    """
    def __init__(self, input_feats, output_feats, n_iter=N_ITER):
        super(PhiSAGESolver, self).__init__()
        self.n_iter = n_iter
        # 内部实例化两个模型
        self.model_real = BuildGCNList(input_feats, output_feats, n_iter)
        self.model_imag = BuildGCNList(input_feats, output_feats, n_iter)
    
    def forward(self, data_list, epoch=None):
        """
        DP 模式下，data_list 是一个列表（原本 Batch 的一部分）。
        我们需要在当前 GPU 上将其 Collate 成一个 Batch，然后跑物理循环。

        Args:
            data_list: 数据列表
            epoch: 当前训练轮次，用于hybrid loss计算（可选）
        """
        # 1. 确保 data_list 中的元素都是 Data 对象
        # DataParallel 可能会传递特殊格式的数据，需要处理
        if not isinstance(data_list, list):
            data_list = [data_list]
        
        # 检查并转换数据格式
        processed_data_list = []
        for item in data_list:
            if isinstance(item, Data):
                # 确保 Data 对象有必要的属性，并且 k_idx 是 tensor
                if not hasattr(item, 'k_idx'):
                    # 如果没有 k_idx，尝试从其他属性获取或设置默认值
                    item.k_idx = torch.tensor([0])
                elif not isinstance(item.k_idx, torch.Tensor):
                    item.k_idx = torch.tensor([item.k_idx] if not isinstance(item.k_idx, (list, tuple)) else item.k_idx)
                processed_data_list.append(item)
            elif isinstance(item, tuple):
                # 如果是 tuple，可能是 (data, ...) 格式，取第一个元素
                if len(item) > 0 and isinstance(item[0], Data):
                    processed_data_list.append(item[0])
                else:
                    raise TypeError(f"无法处理的数据格式: {type(item)}, 内容: {item}")
            else:
                # 尝试直接使用，如果失败会抛出异常
                processed_data_list.append(item)
        
        # 2. 在当前 GPU 上构建 Batch
        # 注意：如果遇到 tupleBatch 错误，可能是 PyG 版本问题
        # 尝试使用 collate 函数作为备选方案
        try:
            batch = Batch.from_data_list(processed_data_list)
        except (AttributeError, TypeError) as e:
            error_msg = str(e)
            if 'stores_as' in error_msg or 'tupleBatch' in error_msg:
                # 使用 collate 函数手动构建 Batch
                from torch_geometric.data.collate import collate
                try:
                    batch, slice_dict, inc_dict = collate(
                        Data,
                        processed_data_list,
                        increment=True,
                        add_batch=True,
                        follow_batch=[],
                    )
                except Exception as e2:
                    # 如果 collate 也失败，提供更详细的错误信息
                    if is_main_process():
                        print(f"❌ Batch.from_data_list 失败: {e}")
                        print(f"❌ collate 也失败: {e2}")
                        print(f"   数据列表长度: {len(processed_data_list)}")
                        print(f"   第一个元素类型: {type(processed_data_list[0]) if processed_data_list else 'None'}")
                        if processed_data_list:
                            print(f"   第一个元素的属性: {dir(processed_data_list[0])}")
                    raise e
            else:
                raise e
        
        # 确保 batch 在正确的设备上
        # 在多卡情况下，DataParallel 会自动处理设备分配，batch 已经在正确的设备上
        # 在单卡情况下，需要确保 batch 在模型所在的设备上
        device = batch.x.device
        
        # 安全地获取模型设备（避免在 DataParallel replica 中出错）
        try:
            # 尝试获取模型参数所在的设备
            model_device = next(self.parameters()).device
            # 如果 batch 不在模型设备上，则移动 batch（主要用于单卡情况）
            if device != model_device:
                batch = batch.to(model_device)
                device = model_device
        except (StopIteration, RuntimeError):
            # 在 DataParallel 的 replica 中，参数可能不可用
            # 此时 batch 已经在正确的设备上（由 DataParallel 保证），直接使用 batch 的设备
            pass
        
        # 2. 准备数据
        k_all = batch.k_idx
        node_batch = batch.batch
        B = k_all.size(0)
        
        # 优化：减少不必要的 clone，使用 view 或直接索引
        # 节点特征：[eps_re, eps_im, r_re, r_im, Ebz_re, Ebz_im, bg_re, bg_im]
        # 根据模型精度决定数据类型 - 兼容不同层类型
        if hasattr(self.model_real.networks[0].gcn1, 'conv'):
            # GCN层的情况
            is_double = self.model_real.networks[0].gcn1.conv.lin_fusion.weight.dtype == torch.float64
        elif hasattr(self.model_real.networks[0].gcn1, 'spatial_conv'):
            # SpectralGCN层的情况
            is_double = self.model_real.networks[0].gcn1.spatial_conv.lin_fusion.weight.dtype == torch.float64
        elif hasattr(self.model_real.networks[0].gcn1, 'linear'):
            # FFTLayer的情况
            is_double = self.model_real.networks[0].gcn1.linear.weight.dtype == torch.float64
        else:
            # 默认情况
            is_double = False

        if is_double:
            eps_feat = batch.x[:, 0:2].double()  # L-BFGS阶段用double
            bg_real = batch.x[:, 6].double()     # 背景场实部 (不随网络更新)
            bg_imag = batch.x[:, 7].double()     # 背景场虚部 (不随网络更新)
            current_E_real = batch.x[:, 4].double()  # 初始电场实部
            current_E_imag = batch.x[:, 5].double()  # 初始电场虚部
        else:
            eps_feat = batch.x[:, 0:2].float()   # Adam阶段用float
            bg_real = batch.x[:, 6].float()      # 背景场实部 (不随网络更新)
            bg_imag = batch.x[:, 7].float()      # 背景场虚部 (不随网络更新)
            current_E_real = batch.x[:, 4].float()  # 初始电场实部
            current_E_imag = batch.x[:, 5].float()  # 初始电场虚部
        
        # 3. 优化：在物理迭代循环之前一次性加载所有矩阵到 GPU
        # 这样可以在多次迭代中重复使用，避免重复的 CPU->GPU 传输（性能关键优化）
        A_list = []
        b_list = []
        # 根据模型精度决定矩阵精度 - 兼容不同层类型
        if hasattr(self.model_real.networks[0].gcn1, 'conv'):
            model_dtype = self.model_real.networks[0].gcn1.conv.lin_fusion.weight.dtype
        elif hasattr(self.model_real.networks[0].gcn1, 'spatial_conv'):
            model_dtype = self.model_real.networks[0].gcn1.spatial_conv.lin_fusion.weight.dtype
        elif hasattr(self.model_real.networks[0].gcn1, 'linear'):
            model_dtype = self.model_real.networks[0].gcn1.linear.weight.dtype
        else:
            model_dtype = torch.float32

        matrix_dtype = torch.complex128 if model_dtype == torch.float64 else torch.complex64

        for b_idx in range(B):
            k = int(k_all[b_idx].item())
            A, b = get_Ab(k, device, matrix_dtype)
            A_list.append(A)
            b_list.append(b)
        
        # 只保存当前迭代的结果，而不是所有历史
        E_real_cur = current_E_real
        E_imag_cur = current_E_imag
        
        # 4. 物理迭代循环
        for iter_idx in range(self.n_iter):
            r_real_list = []
            r_imag_list = []
            
            # 计算残差 r = b - A*E
            # 优化：使用已加载的矩阵（已在 GPU 上），避免重复传输
            with torch.no_grad():
                for b_idx in range(B):
                    mask = (node_batch == b_idx)
                    E_r = E_real_cur[mask]
                    E_i = E_imag_cur[mask]
                    
                    # 使用已加载的矩阵（已在 GPU 上，无需重复传输）
                    A = A_list[b_idx]
                    b_vec = b_list[b_idx]
                    
                    E_c = torch.complex(E_r, E_i)
                    Ax = torch.sparse.mm(A, E_c.unsqueeze(-1)).squeeze(-1)
                    r_c = b_vec - Ax
                    
                    # 根据当前模型精度决定数据类型 - 兼容不同层类型
                    if hasattr(self.model_real.networks[0].gcn1, 'conv'):
                        use_double = self.model_real.networks[0].gcn1.conv.lin_fusion.weight.dtype == torch.float64
                    elif hasattr(self.model_real.networks[0].gcn1, 'spatial_conv'):
                        use_double = self.model_real.networks[0].gcn1.spatial_conv.lin_fusion.weight.dtype == torch.float64
                    elif hasattr(self.model_real.networks[0].gcn1, 'linear'):
                        use_double = self.model_real.networks[0].gcn1.linear.weight.dtype == torch.float64
                    else:
                        use_double = False

                    if use_double:
                        r_real_list.append(r_c.real.double())
                        r_imag_list.append(r_c.imag.double())
                    else:
                        r_real_list.append(r_c.real.float())
                        r_imag_list.append(r_c.imag.float())
                    
                    # 释放中间变量（但保留矩阵 A 和 b，因为还要在下次迭代中使用）
                    del E_c, Ax, r_c
            
            r_real = torch.cat(r_real_list, dim=0)
            r_imag = torch.cat(r_imag_list, dim=0)
            
            # 优化：释放中间列表
            del r_real_list, r_imag_list
            
            # 构造输入（优化：使用 view 而不是 unsqueeze，节省显存）
            # 节点特征包含：[eps, r, E_current, bg] 共8个通道
            x_in = torch.cat([
                eps_feat,                # eps_real, eps_imag [N, 2]
                r_real.view(-1, 1),      # r_real [N, 1]
                r_imag.view(-1, 1),      # r_imag [N, 1]
                E_real_cur.view(-1, 1),  # 当前电场实部 [N, 1] (随迭代更新)
                E_imag_cur.view(-1, 1),  # 当前电场虚部 [N, 1] (随迭代更新)
                bg_real.view(-1, 1),     # 背景场实部 [N, 1] (不随网络更新)
                bg_imag.view(-1, 1)      # 背景场虚部 [N, 1] (不随网络更新)
            ], dim=1)
            
            # 优化：使用梯度检查点节省显存（在训练模式下）
            use_checkpoint = USE_GRADIENT_CHECKPOINTING

            # 使用梯度检查点
            if self.training and use_checkpoint:
                # 梯度检查点：在前向传播时不保存中间激活值，反向传播时重新计算
                # 这会节省显存，但会增加计算时间（约20-30%）
                def gcn_forward_real(x, edge_index, batch, iter_idx):
                    return self.model_real(x, edge_index, batch, iter_idx)
                def gcn_forward_imag(x, edge_index, batch, iter_idx):
                    return self.model_imag(x, edge_index, batch, iter_idx)

                delta_real = checkpoint(gcn_forward_real, x_in, batch.edge_index, batch.batch, iter_idx, use_reentrant=False)
                delta_imag = checkpoint(gcn_forward_imag, x_in, batch.edge_index, batch.batch, iter_idx, use_reentrant=False)
            else:
                # 正常前向传播
                delta_real = self.model_real(x_in, batch.edge_index, batch.batch, iter_idx)
                delta_imag = self.model_imag(x_in, batch.edge_index, batch.batch, iter_idx)
            
            # 展平并更新（使用 in-place 操作节省显存）
            delta_real = delta_real.view(-1)
            delta_imag = delta_imag.view(-1)
            
            # 优化：直接更新，不保存历史（只保留当前值）
            E_real_next = E_real_cur + delta_real
            E_imag_next = E_imag_cur + delta_imag
            
            # 优化：释放中间变量
            del x_in, delta_real, delta_imag, r_real, r_imag
            
            # 更新当前值（为下一次迭代准备）
            E_real_cur = E_real_next
            E_imag_cur = E_imag_next
            
            # 优化：不在每次迭代中清理显存，避免阻塞
            # 仅在阶段切换时清理显存碎片
        
        # 5. 计算损失
        num_nodes = batch.x.size(0)  # 总节点数
        if LOSS_TYPE == "hybrid":
            # compute_hybrid_loss现在返回三个值，避免重复计算
            loss_sum, mse_loss_sum, phi_loss_sum = compute_hybrid_loss(E_real_cur, E_imag_cur, batch.y, k_all, node_batch, B, device, matrix_dtype, epoch, num_nodes)
        else:
            # 非hybrid模式下，只计算实际需要的loss
            if LOSS_TYPE == "phi":
                loss_sum = compute_phi_loss(E_real_cur, E_imag_cur, k_all, node_batch, B, device, matrix_dtype, num_nodes)
            elif LOSS_TYPE == "asinh":
                loss_sum = compute_asinh_loss(E_real_cur, E_imag_cur, batch.y, num_nodes)
            else:
                # 默认使用MSE loss
                loss_sum = compute_mse_loss(E_real_cur, E_imag_cur, batch.y, num_nodes)

            # 非hybrid模式下不需要单独的MSE和Phi loss，使用占位符
            mse_loss_sum = torch.tensor(0.0, device=device)
            phi_loss_sum = torch.tensor(0.0, device=device)

        # 6. 计算相对误差形式的RES loss
        # RES loss需要真实标签，用于计算相对误差
        true_real = batch.y[:, 0]
        true_imag = batch.y[:, 1]
        # RES loss = ||x^FEM - x^GCN||_2^2 / ||x^FEM||_2^2
        # 其中 x^FEM 是真实解，x^GCN 是预测解
        # ||x||_2^2 = sum_i |x_i|^2，计算平方的相对误差（与公式一致）
        res_loss_sum = torch.tensor(0.0, device=device, dtype=loss_sum.dtype)
        with torch.no_grad():
            for b_idx in range(B):
                mask = (node_batch == b_idx)
                # 预测解 x^GCN
                pred_real = E_real_cur[mask]
                pred_imag = E_imag_cur[mask]
                # 真实解 x^FEM
                true_r = true_real[mask]
                true_i = true_imag[mask]
                
                # 计算 ||x^FEM - x^GCN||_2^2 = sum_i |true_i - pred_i|^2
                # 对于复数向量，需要分别计算实部和虚部
                diff_real = true_r - pred_real
                diff_imag = true_i - pred_imag
                # 提前取绝对值再平方
                numerator = (torch.abs(diff_real).pow(2).sum() + torch.abs(diff_imag).pow(2).sum())
                
                # 计算 ||x^FEM||_2^2 = sum_i |true_i|^2
                # 提前取绝对值再平方
                denominator = (torch.abs(true_r).pow(2).sum() + torch.abs(true_i).pow(2).sum())
                
                # 避免除零，如果分母太小则使用一个小的epsilon
                epsilon = 1e-10
                if denominator > epsilon:
                    # 平方的相对误差：||x^FEM - x^GCN||_2^2 / ||x^FEM||_2^2（与公式一致）
                    rel_error_squared = numerator / denominator
                    res_loss_sum = res_loss_sum + rel_error_squared
                else:
                    # 如果真实解范数太小，使用绝对误差的平方
                    res_loss_sum = res_loss_sum + numerator
                
                # 释放中间变量
                del diff_real, diff_imag, pred_real, pred_imag, true_r, true_i
        
        # 优化：在迭代结束后释放矩阵列表（释放显存）
        del A_list, b_list
        
        # 释放变量
        del E_real_cur, E_imag_cur, true_real, true_imag
        
        # 注意：PyG DataParallel 可能不支持 tuple 返回值，会尝试将其当作 Batch 处理
        # 因此返回单个 tensor：[loss_sum, res_loss_sum, num_nodes, num_samples, mse_loss_sum, phi_loss_sum]
        # 这样可以避免 'tupleBatch' 错误
        # num_nodes: 总节点数（用于loss的平均）
        # num_samples: 总样本数（用于RES loss的平均，因为相对误差是针对每个样本的）
        num_nodes = batch.x.size(0)  # 总节点数（从 batch.x 获取，不需要从 E_real_cur）
        num_samples = B  # 样本数（batch中的图数量）
        num_nodes_tensor = torch.tensor(num_nodes, dtype=torch.float32, device=device)
        num_samples_tensor = torch.tensor(num_samples, dtype=torch.float32, device=device)
        mse_loss_tensor = mse_loss_sum.to(dtype=torch.float32, device=device) if hasattr(mse_loss_sum, 'to') else torch.tensor(float(mse_loss_sum), dtype=torch.float32, device=device)
        phi_loss_tensor = phi_loss_sum.to(dtype=torch.float32, device=device) if hasattr(phi_loss_sum, 'to') else torch.tensor(float(phi_loss_sum), dtype=torch.float32, device=device)
        # 返回形状为 [6] 的 tensor: [loss_sum, res_loss_sum, num_nodes, num_samples, mse_loss_sum, phi_loss_sum]
        return torch.stack([loss_sum, res_loss_sum, num_nodes_tensor, num_samples_tensor, mse_loss_tensor, phi_loss_tensor])



# ==========================================
# 5. 主程序
# ==========================================
def setup_ddp(rank, world_size):
    """设置DDP环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = MASTER_PORT

    # 设置NCCL优化环境变量，提高分布式训练稳定性
    os.environ['NCCL_TIMEOUT'] = '1800000'     # 30分钟超时 (毫秒)
    os.environ['NCCL_IB_DISABLE'] = '1'         # 禁用IB以提高兼容性
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'     # 使用本地环回接口
    os.environ['NCCL_DEBUG'] = 'WARN'           # 设置调试级别

    # 初始化进程组，设置更长的超时时间（30分钟）以避免NCCL超时
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))

    # 设置当前进程的GPU
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    return device

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

def main_worker(rank, world_size):
    """DDP训练的主工作函数"""
    device = setup_ddp(rank, world_size)

    # 只有主进程(rank 0)输出信息
    is_main_process = (rank == 0)

    # DDP环境中，每个进程只负责一个GPU
    device_ids = [rank]
    
    if is_main_process:
        print(f"🚀 启动 DDP 分布式训练")
        print(f"   进程 {rank}/{world_size}")
        print(f"   GPU: {device}")
        print(f"   每个GPU的BatchSize: {TOTAL_BATCH_SIZE}")
        print(f"   总BatchSize: {TOTAL_BATCH_SIZE * world_size}")
        print(f"   NCCL超时时间: 30分钟")
        print(f"   环境变量: NCCL_TIMEOUT=1800000ms")

        # 打印数据集配置信息（只在主进程中打印）
        from config import DATASET_TYPE, DATA_ROOT_PATH, SCA_PREFIX
        print(f"📊 全局数据集配置: {DATASET_TYPE}")
        print(f"   数据根目录: {DATA_ROOT_PATH}")
        print(f"   文件命名前缀: {SCA_PREFIX}")
    
    # 2. 数据准备
    data_mapping, n_total = scan_all_data(root_data_path)
    
    # 优化：使用多进程并行构建数据集（如果数据量大）
    dataset = []

    # 获取所有有效的 k 索引
    valid_k_list = [k for k in range(1, n_total + 1) if k in data_mapping]

    # 使用串行方式构建数据集（多进程会导致共享内存问题）
    # 注意：多进程传递大量 Data 对象时，Python 的 multiprocessing 使用共享内存
    # 可能导致 "Too many open files" 错误，因此使用串行方式更稳定
    if is_main_process:
        print("构建数据集...")
        print("   串行构建数据集（稳定可靠）...")
    for idx, k in enumerate(valid_k_list):
        try:
            data = build_graph_data(k)
            if data is not None:
                dataset.append(data)
            # 每 100 个样本显示一次进度
            if (idx + 1) % 1000 == 0 and is_main_process:
                print(f"   进度: {idx + 1}/{len(valid_k_list)} ({100*(idx+1)/len(valid_k_list):.1f}%)")
        except Exception as e:
            if (idx + 1) % 1000 == 0 and is_main_process:  # 只在显示进度时打印错误
                print(f"   构建样本 {k} 失败: {e}")
            pass
    
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，无法进行训练！")

    if is_main_process:
        print(f"   数据集大小: {len(dataset)}")
    train_ds, test_ds = train_test_split(dataset, test_size=0.2, random_state=42)

    # # 定义测试集的编号（从1开始计数，对应数据索引）
    # test_indices = [1,13,14,18,19,24,25,36,37,42,43,48,49,60,61,66,67,72,73,84,85,90,91,96]

    # # 根据编号划分训练集和测试集
    # train_ds = []
    # test_ds = []

    # for i, data in enumerate(dataset):
    #     # data.k_idx 存储的是数据的编号（从1开始）
    #     data_idx = data.k_idx.item()
    #     if data_idx in test_indices:
    #         test_ds.append(data)
    #     else:
    #         train_ds.append(data)

    if is_main_process:
        print(f"   训练集: {len(train_ds)}, 测试集: {len(test_ds)}")
    # print(f"   测试集编号: {test_indices}")
    
    # 【关键】使用 DataListLoader
    # 注意：PyG 的 DataListLoader 可能不支持 num_workers（专为 DataParallel 设计）
    # 但我们可以尝试设置，如果不支持会自动忽略
    # 优化：添加 pin_memory 以加速数据传输到GPU

    # Adam阶段：开启shuffle增加训练随机性，提高泛化能力
    train_loader = DataListLoader(
    train_ds,
    batch_size=TOTAL_BATCH_SIZE,
    shuffle=True,  # Adam阶段开启shuffle
    drop_last=True,
    pin_memory=True,  # 确保为True
    num_workers=NUM_WORKERS,  # 添加这行！关键优化
    persistent_workers=True if NUM_WORKERS > 0 else False  # 保持worker进程
)

    # 测试阶段：关闭shuffle，确保评估结果一致性
    test_loader = DataListLoader(
    test_ds,
    batch_size=TOTAL_BATCH_SIZE,
    shuffle=False,  # 测试阶段关闭shuffle
    pin_memory=True,
    num_workers=NUM_WORKERS,  # 添加
    persistent_workers=True if NUM_WORKERS > 0 else False
)
    
    # 3. 预加载矩阵到所有 GPU
    load_matrix_to_cache(data_mapping, n_total, device_ids, is_main_process)
    
    # 【关键】DDP同步点：确保所有进程都完成矩阵预加载后再继续
    # 这可以避免进程间不同步导致的NCCL心跳超时
    if dist.is_initialized():
        dist.barrier()
        if is_main_process:
            print("✅ 所有进程已完成矩阵预加载，继续训练...")
    
    # 4. 模型初始化与 DP 包装
    # Adam阶段使用float32加快速度，L-BFGS阶段切换到float64提高精度
    # 节点特征包含：[eps, r, E_current, bg] 共8个通道
    solver = PhiSAGESolver(input_feats=8, output_feats=1, n_iter=N_ITER).float()  # Adam阶段用float32
    solver.to(device)
    
    # 优化：使用 torch.compile 加速（如果支持）
    # 注意：torch.compile 需要在 DataParallel 包装之前应用
    # 使用局部变量来避免修改全局变量
    use_compile = USE_COMPILE
    if use_compile and is_main_process:
        try:
            # 检查 PyTorch 版本是否支持 compile
            if hasattr(torch, 'compile'):
                print("✅ 启用 torch.compile 优化...")
                print("   ⚠️  注意：首次运行需要编译时间，可能较慢")
                print("   ⚠️  如果遇到错误，请将 USE_COMPILE 设置为 False")
                # 使用 'reduce-overhead' 模式，适合多次调用的场景
                solver = torch.compile(solver, mode='reduce-overhead')
            else:
                print("⚠️  PyTorch 版本不支持 torch.compile，跳过此优化")
                use_compile = False
        except Exception as e:
            print(f"⚠️  torch.compile 失败: {e}，继续使用未编译版本")
            print(f"   💡 提示：如果遇到兼容性问题，请将 USE_COMPILE 设置为 False")
            use_compile = False
    
    # 打印网络维度信息
    if is_main_process:
        from config import NETWORK_USE_CUSTOM_DIMS, NETWORK_BASE_DIM, NETWORK_CUSTOM_DIMS, NETWORK_POOL_RATIOS

        print("\n🔍 网络结构维度信息:")
        print(f"   迭代次数 (n_iter): {solver.n_iter}")
        print(f"   输入特征数: {solver.model_real.input_feats}")
        print(f"   输出特征数: {solver.model_real.output_feats}")

        # 显示配置来源
        print(f"   配置来源: config.py")
        if NETWORK_USE_CUSTOM_DIMS:
            print(f"     • 使用自定义维度: {NETWORK_CUSTOM_DIMS}")
        else:
            print(f"     • 使用基础维度: {NETWORK_BASE_DIM} (自动计算: [{NETWORK_BASE_DIM}, {NETWORK_BASE_DIM*2}, {NETWORK_BASE_DIM*4}])")
        print(f"     • 池化配置: {NETWORK_POOL_RATIOS}")

        try:
            # 获取第一个网络来显示维度信息
            first_network = solver.model_real.networks[0]

            # 显示网络维度配置
            if hasattr(first_network, 'gcn1') and hasattr(first_network.gcn1, 'conv') and hasattr(first_network.gcn1.conv, 'lin_fusion'):
                gcn1_weight = first_network.gcn1.conv.lin_fusion.weight
                gcn2_weight = first_network.gcn2.conv.lin_fusion.weight
                gcn3_weight = first_network.gcn3.conv.lin_fusion.weight

                print(f"   GCN层实际维度:")
                print(f"     • gcn1: {gcn1_weight.shape[1]} → {gcn1_weight.shape[0]}")
                print(f"     • gcn2: {gcn2_weight.shape[1]} → {gcn2_weight.shape[0]}")
                print(f"     • gcn3: {gcn3_weight.shape[1]} → {gcn3_weight.shape[0]}")
                print(f"   网络架构: U-Net风格 (编码器-解码器)")

                # 计算参数量
                total_params = sum(p.numel() for p in solver.parameters())
                real_params = sum(p.numel() for p in solver.model_real.parameters())
                imag_params = sum(p.numel() for p in solver.model_imag.parameters())

                print(f"   参数统计:")
                print(f"     • 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
                print(f"     • Real网络: {real_params:,} 参数")
                print(f"     • Imag网络: {imag_params:,} 参数")
                print(f"     • 单迭代网络: {real_params // solver.n_iter:,} 参数")
            else:
                print("   ⚠️  无法获取详细的网络维度信息")

        except Exception as e:
            print(f"   ⚠️  获取网络维度信息时出错: {e}")
            # 仍然显示基本参数量信息
            total_params = sum(p.numel() for p in solver.parameters())
            print(f"   总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # DDP包装
    solver = DDP(solver, device_ids=[rank], output_device=rank)
    if is_main_process:
        print("✅ 模型已通过 DDP 包装")
    
    # 使用纯float32训练，无需GradScaler
    
    # 5. 优化器
    # 注意：DP 包装后，参数名会多出 .module 前缀，但不影响 optimizer 识别
    # DDP优化：学习率按GPU数量线性缩放（总batch_size增大）
    ddp_lr = LR   # 每个GPU的基础学习率乘以GPU数量
    optimizer_adam = optim.Adam(solver.parameters(), lr=ddp_lr)

    # 使用ReduceLROnPlateau调度器，基于验证损失自动调整学习率
    scheduler = ReduceLROnPlateau(
        optimizer_adam,
        mode=REDUCE_LR_MODE,
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=REDUCE_LR_MIN_LR
    )
    
    # 6. 训练循环
    # 初始化训练状态变量
    best_loss = float('inf')  # 用于早停判断的最佳loss
    best_epoch = -1
    best_saved_loss = float('inf')  # 用于保存模型判断的最佳loss（hybrid模式下200epoch后开始）

    # 早停机制变量
    early_stopping_counter = 0
    early_stopping_best_loss = float('inf')

    # 用于记录loss变化曲线
    train_losses = []
    test_losses = []
    train_mse_losses = []
    test_mse_losses = []
    train_res_losses = []
    test_res_losses = []

    # 训练数据保存
    training_data = {
        'epochs': [],
        'train_losses': [],
        'test_losses': [],
        'train_mse_losses': [],
        'test_mse_losses': [],
        'train_res_losses': [],
        'test_res_losses': [],
        'train_relative_errors': [],
        'test_relative_errors': []
    }

    # Hybrid loss模式下的MSE和Phi loss记录
    if LOSS_TYPE == "hybrid":
        hybrid_loss_data = {
            'epochs': [],
            'train_mse_losses': [],
            'train_phi_losses': [],
            'test_mse_losses': [],
            'test_phi_losses': []
        }

    # 记录训练总开始时间和上次打印时间
    total_start_time = time.time()
    last_print_time = total_start_time
    
    # 加载预训练模型（如果启用）
    if LOAD_PRETRAINED_MODEL:
        pretrained_dir = PRETRAINED_MODEL_DIR if PRETRAINED_MODEL_DIR is not None else SAVE_DIR
        load_success = load_pretrained_model(pretrained_dir, solver, N_ITER, device, is_main_process)
        if load_success and is_main_process:
            print("🔄 已加载预训练模型权重，优化器等状态已重新初始化")
            print("   将从epoch 0开始训练（在新数据集上）")
    
    for epoch in range(TOTAL_EPOCHS):
        solver.train()
        optimizer = optimizer_adam  # 只使用Adam优化器
        
        epoch_loss_sum = 0.0
        epoch_mse_sum = 0.0
        epoch_res_sum = 0.0
        epoch_phi_sum = 0.0  # Hybrid loss模式下的Phi loss累加器
        total_nodes = 0
        total_samples = 0  # 用于RES loss的平均（相对误差是针对每个样本的）
        
        # ==========================
        # Adam Training (纯float32)
        # ==========================
        for data_list in train_loader:
                optimizer.zero_grad()

                # Forward: list -> (split) -> GPUs -> (run) -> (gather) -> results
                outputs = solver(data_list, epoch)

                # 使用辅助函数提取 loss 和 num_nodes
                batch_loss_sum, batch_res_sum, num_nodes, num_samples, batch_mse_sum, batch_phi_sum = extract_loss_and_num_nodes(outputs)

                # 计算平均 Loss（Phi和Asinh loss不做节点平均，MSE loss使用节点平均）
                # Hybrid loss根据当前权重决定是否节点平均
                if num_nodes == 0:
                    continue  # 跳过空批次
                if LOSS_TYPE == "mse":
                    loss_mean = batch_loss_sum / num_nodes
                else:
                    loss_mean = batch_loss_sum

                loss_mean.backward()
                optimizer.step()
                
                epoch_loss_sum += batch_loss_sum.item()
                epoch_res_sum += batch_res_sum.item()
                # 只在hybrid模式下累加额外的loss分量
                if LOSS_TYPE == "hybrid":
                    epoch_mse_sum += batch_mse_sum.item()
                    epoch_phi_sum += batch_phi_sum.item()
                total_nodes += num_nodes
                total_samples += num_samples

        # 优化：每10个epoch清理一次显存
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            

        # ==========================
        # 日志与测试
        # ==========================
        num_batches = len(train_loader)
        if num_batches > 0:
            # 计算平均loss
            if LOSS_TYPE == "mse":
                # MSE loss: 除以总节点数，得到每个节点的平均loss
                avg_train_loss = epoch_loss_sum / total_nodes if total_nodes > 0 else 0.0
                avg_train_mse = avg_train_loss  # MSE模式下MSE loss就是总loss
            elif LOSS_TYPE == "hybrid":
                # Hybrid loss: 除以batch数，得到每个batch的平均loss
                avg_train_loss = epoch_loss_sum / num_batches
                avg_train_mse = epoch_mse_sum / total_nodes if total_nodes > 0 else 0.0
            else:
                # Phi/Asinh loss: 除以batch数，得到每个batch的平均loss
                avg_train_loss = epoch_loss_sum / num_batches
                avg_train_mse = 0.0  # 非hybrid模式下不计算MSE loss
        else:
            avg_train_loss = 0.0
            avg_train_mse = 0.0

        # RES Loss 是 Sum of Relative Errors，所以除以 total_samples (总图数) 是对的
        avg_train_res = epoch_res_sum / total_samples if total_samples > 0 else 0.0

        # 计算平均Phi loss（在hybrid模式下）
        if LOSS_TYPE == "hybrid":
            avg_train_phi = epoch_phi_sum / total_nodes if total_nodes > 0 else 0.0
        else:
            avg_train_phi = 0.0
        
        # ==========================
        # 测试集评估 (同理修正)
        # ==========================
        solver.eval()
        test_loss_sum = 0.0
        test_mse_sum = 0.0
        test_res_sum = 0.0
        test_phi_sum = 0.0  # Hybrid loss模式下的Phi loss累加器
        test_total_nodes = 0
        test_total_samples = 0
        test_num_batches = len(test_loader)

        with torch.no_grad():
            # 评估阶段（纯float32）
            for data_list in test_loader:
                outputs = solver(data_list, epoch)
                batch_loss_sum, batch_res_sum, num_nodes, num_samples, batch_mse_sum, batch_phi_sum = extract_loss_and_num_nodes(outputs)
                
                if num_nodes == 0:
                    continue
                
                # 累加基本loss
                test_res_sum += batch_res_sum.item()  # 累加RES loss
                test_loss_sum += batch_loss_sum.item() * num_nodes  # 转换为总损失再累加

                # 只在hybrid模式下累加额外的loss分量
                if LOSS_TYPE == "hybrid":
                    test_mse_sum += batch_mse_sum.item()  # 已经是总损失
                    test_phi_sum += batch_phi_sum.item()  # 已经是总损失
                test_total_nodes += num_nodes
                test_total_samples += num_samples
                test_num_batches += 1
        
        if test_num_batches > 0:
            # 按总节点数平均loss
            if LOSS_TYPE == "mse":
                # MSE loss: 除以总节点数，得到每个节点的平均loss
                avg_test_loss = test_loss_sum / test_total_nodes if test_total_nodes > 0 else 0.0
                avg_test_mse = avg_test_loss  # MSE模式下MSE loss就是总loss
            elif LOSS_TYPE == "hybrid":
                # Hybrid loss: 除以总节点数
                avg_test_loss = test_loss_sum / test_total_nodes if test_total_nodes > 0 else 0.0
                avg_test_mse = test_mse_sum / test_total_nodes if test_total_nodes > 0 else 0.0
            else:
                # Phi/Asinh loss: 除以总节点数
                avg_test_loss = test_loss_sum / test_total_nodes if test_total_nodes > 0 else 0.0
                avg_test_mse = 0.0  # 非hybrid模式下不计算MSE loss
        else:
            avg_test_loss = 0.0
            avg_test_mse = 0.0
            
        avg_test_res = test_res_sum / test_total_samples if test_total_samples > 0 else 0.0

        # 计算平均Phi loss（在hybrid模式下）
        if LOSS_TYPE == "hybrid":
            avg_test_phi = test_phi_sum / test_total_nodes if test_total_nodes > 0 else 0.0
        else:
            avg_test_phi = 0.0

        # 使用ReduceLROnPlateau调度器
        scheduler.step(avg_test_loss)
        
        # Hybrid模式下从一开始就记录最佳loss并保存模型
        if LOSS_TYPE == "hybrid":
            if avg_test_loss < best_saved_loss:
                best_saved_loss = avg_test_loss
                best_epoch = epoch

                # 保存模型
                raw_model = solver.module if isinstance(solver, (DataParallel, DDP)) else solver
                os.makedirs(SAVE_DIR, exist_ok=True)

                # 保存模型权重
                n_iter = raw_model.n_iter
                for i in range(n_iter):
                    torch.save(raw_model.model_real.get_network(i).state_dict(),
                               os.path.join(SAVE_DIR, f"real_iter_{i}.pth"))
                    torch.save(raw_model.model_imag.get_network(i).state_dict(),
                               os.path.join(SAVE_DIR, f"imag_iter_{i}.pth"))
        else:
            # 非hybrid模式，正常保存逻辑
            if avg_test_loss < best_saved_loss:
                best_saved_loss = avg_test_loss
                best_epoch = epoch

                # 保存模型
                raw_model = solver.module if isinstance(solver, (DataParallel, DDP)) else solver
                os.makedirs(SAVE_DIR, exist_ok=True)

                # 保存模型权重
                n_iter = raw_model.n_iter
                for i in range(n_iter):
                    torch.save(raw_model.model_real.get_network(i).state_dict(),
                               os.path.join(SAVE_DIR, f"real_iter_{i}.pth"))
                    torch.save(raw_model.model_imag.get_network(i).state_dict(),
                               os.path.join(SAVE_DIR, f"imag_iter_{i}.pth"))
        
        # 记录loss用于绘图
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_mse_losses.append(avg_train_mse)
        test_mse_losses.append(avg_test_mse)
        train_res_losses.append(avg_train_res)
        test_res_losses.append(avg_test_res)

        # 保存到训练数据字典
        training_data['epochs'].append(epoch)
        training_data['train_losses'].append(avg_train_loss)
        training_data['test_losses'].append(avg_test_loss)
        training_data['train_mse_losses'].append(avg_train_mse)
        training_data['test_mse_losses'].append(avg_test_mse)
        training_data['train_res_losses'].append(avg_train_res)
        training_data['test_res_losses'].append(avg_test_res)

        # 保存hybrid loss数据
        if LOSS_TYPE == "hybrid":
            hybrid_loss_data['epochs'].append(epoch)
            hybrid_loss_data['train_mse_losses'].append(avg_train_mse)
            hybrid_loss_data['train_phi_losses'].append(avg_train_phi)
            hybrid_loss_data['test_mse_losses'].append(avg_test_mse)
            hybrid_loss_data['test_phi_losses'].append(avg_test_phi)

        # 全局早停检查（针对所有阶段）
        if epoch >= EARLY_STOPPING_START_EPOCH:
            early_stopping_enabled = True
            if avg_test_loss < early_stopping_best_loss - EARLY_STOPPING_MIN_DELTA:
                # 有显著改善，重置计数器
                early_stopping_best_loss = avg_test_loss
                early_stopping_counter = 0
            else:
                # 无显著改善，计数器加1
                early_stopping_counter += 1

            # 检查是否达到早停条件
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                if is_main_process:
                    print(f"🛑 全局早停激活！")
                    print(f"   连续{EARLY_STOPPING_PATIENCE}个epoch无显著改善")
                    print(f"   最小改善阈值: {EARLY_STOPPING_MIN_DELTA:.0e}")
                    print(f"   当前loss: {avg_test_loss:.6e}")
                    print(f"   最佳loss: {early_stopping_best_loss:.6e} (epoch {best_epoch})")
                break
        else:
            # 预热阶段，跟踪最佳loss但不触发早停
            if avg_test_loss < early_stopping_best_loss:
                early_stopping_best_loss = avg_test_loss


        
        if epoch % EPOCH_PRINT == 0 and is_main_process:
            current_time = time.time()
            interval_time = current_time - last_print_time
            last_print_time = current_time

            # 为hybrid loss添加权重信息
            loss_info = f"Train Loss: {avg_train_loss:.6e} | Test Loss: {avg_test_loss:.6e}"

            print(f"Epoch {epoch:4d} | {loss_info} | "
                  f"Train RelErr: {avg_train_res:.6e} | Test RelErr: {avg_test_res:.6e} | "
                  f"Interval: {interval_time:.1f}s")
    
    # 训练结束，输出总耗时
    if is_main_process:
        total_time = time.time() - total_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"\n✅ 训练完成！总耗时: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time:.1f}秒)")

        # 输出最佳模型信息
        # 检查是否有模型被保存（通过best_epoch或best_saved_loss判断）
        if best_epoch >= 0 and best_saved_loss < float('inf'):
            print(f"\n🏆 最佳模型:")
            print(f"   🎯 Epoch: {best_epoch}")
            print(f"   📊 Test Loss: {best_saved_loss:.6e}")
            print(f"   💾 模型已保存到: {SAVE_DIR}")
        else:
            print("⚠️  警告：未找到有效的模型（可能训练失败）")
            print(f"   📂 模型保存目录: {SAVE_DIR}")

    # 只在主进程中保存结果和生成图表
    if is_main_process:
        # 保存训练数据
        training_data_path = os.path.join(OUTPUT_DIR, "training_data.json")
        save_training_data(training_data, training_data_path)

        # 生成训练曲线
        curve_pdf_path = os.path.join(OUTPUT_DIR, "training_curve.pdf")
        curve_svg_path = os.path.join(OUTPUT_DIR, "training_curve.svg")
        plot_training_curve(train_losses, test_losses, curve_pdf_path)

        # 生成MSE和RES loss曲线
        mse_res_path = os.path.join(OUTPUT_DIR, "mse_res_loss.svg")
        plot_mse_res_loss(train_mse_losses, test_mse_losses, train_res_losses, test_res_losses, mse_res_path)

        # 保存hybrid loss的MSE和Phi loss到txt文件（每50轮保存一次）
        if LOSS_TYPE == "hybrid":
            hybrid_loss_file = os.path.join(OUTPUT_DIR, "hybrid_loss_components.txt")
            with open(hybrid_loss_file, 'w') as f:
                f.write("Hybrid Loss Components (MSE + 1*Phi) - Every 50 epochs\n")
                f.write("="*60 + "\n")
                f.write("Epoch\tTrain_MSE\tTrain_Phi\tTest_MSE\tTest_Phi\n")
                # 每50轮保存一次
                for i, epoch in enumerate(hybrid_loss_data['epochs']):
                    if epoch % 50 == 0:  # 每50轮保存一次
                        f.write(f"{epoch}\t{hybrid_loss_data['train_mse_losses'][i]:.6e}\t")
                        f.write(f"{hybrid_loss_data['train_phi_losses'][i]:.6e}\t")
                        f.write(f"{hybrid_loss_data['test_mse_losses'][i]:.6e}\t")
                        f.write(f"{hybrid_loss_data['test_phi_losses'][i]:.6e}\n")

            print(f"✅ Hybrid loss分量已保存到: {hybrid_loss_file}")

    # 计算相对误差分布
    if is_main_process:
        print("\n正在计算相对误差分布...")
    raw_model = solver.module if isinstance(solver, (DataParallel, DDP)) else solver
    # 根据模型精度决定矩阵精度 - 兼容不同层类型
    if hasattr(raw_model.model_real.networks[0].gcn1, 'conv'):
        model_precision = raw_model.model_real.networks[0].gcn1.conv.lin_fusion.weight.dtype
    elif hasattr(raw_model.model_real.networks[0].gcn1, 'spatial_conv'):
        model_precision = raw_model.model_real.networks[0].gcn1.spatial_conv.lin_fusion.weight.dtype
    elif hasattr(raw_model.model_real.networks[0].gcn1, 'linear'):
        model_precision = raw_model.model_real.networks[0].gcn1.linear.weight.dtype
    else:
        model_precision = torch.float32

    matrix_dtype = torch.complex128 if model_precision == torch.float64 else torch.complex64

    train_relative_errors = compute_relative_errors(solver, train_loader, data_mapping, device, matrix_dtype)
    test_relative_errors = compute_relative_errors(solver, test_loader, data_mapping, device, matrix_dtype)

    # 保存相对误差数据
    training_data['train_relative_errors'] = train_relative_errors
    training_data['test_relative_errors'] = test_relative_errors

    # 计算MSE loss分布（使用训练过程中的MSE loss值）
    if is_main_process:
        print("\n正在生成MSE loss分布图...")
    train_mse_samples = [loss for loss in train_mse_losses for _ in range(10)]  # 重复值以获得更好的分布
    test_mse_samples = [loss for loss in test_mse_losses for _ in range(10)]    # 重复值以获得更好的分布

    # 生成MSE loss分布图
    mse_dist_path = os.path.join(OUTPUT_DIR, "mse_loss_distribution.svg")
    plot_mse_loss_distribution(train_mse_samples, test_mse_samples, mse_dist_path)

def main():
    """主函数：启动DDP训练"""
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA GPU，无法进行DDP训练")
        return

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("❌ 未检测到任何GPU")
        return

    print(f"🚀 启动DDP训练，使用 {world_size} 个GPU")

    # 使用spawn方式启动多进程
    try:
        mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"❌ DDP训练失败: {e}")
        raise

if __name__ == "__main__":
    main()