import torch
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
import os
import glob
import re

# ================= 配置区域 =================
# 从全局配置文件导入数据集配置
from config import DATASET_TYPE, DATA_ROOT_PATH, SCA_PREFIX, DATASET_DIRS_PATTERN

# 根据全局配置设置路径
root_data_path = os.path.join(os.path.dirname(__file__), DATA_ROOT_PATH)
sca_prefix = SCA_PREFIX
dataset_dirs_pattern = DATASET_DIRS_PATTERN
# ===========================================

def is_main_process():
    """检查是否是主进程（rank 0）"""
    rank = int(os.environ.get("RANK", "0"))
    return rank == 0

# 全局变量
data_mapping = {}
n_total = 0

# 移除模块级别的打印，避免DDP重复打印
# 配置信息将在实际使用时打印
# ===========================================

def scan_all_data(root_path):
    """
    扫描A-4目录下的所有四组数据（A-TainDataset, A-TainDataset2, A-TainDataset3, A-TainDataset4）
    依然以 edge 文件为锚点
    """
    global data_mapping, n_total

    # 如果已经扫描过，直接返回，避免重复扫描
    if data_mapping and n_total > 0:
        return data_mapping, n_total

    data_mapping = {}
    k_idx = 1

    # 获取所有数据子目录
    dataset_dirs = [d for d in os.listdir(root_path)
                   if os.path.isdir(os.path.join(root_path, d)) and d.startswith(dataset_dirs_pattern.replace('*', ''))]

    if is_main_process():
        print(f"发现 {len(dataset_dirs)} 个数据集目录: {dataset_dirs}")

    total_folders = 0
    for dataset_dir in sorted(dataset_dirs):
        dataset_path = os.path.join(root_path, dataset_dir)

        # 扫描每个数据集目录中的sca*_data文件夹
        all_items = os.listdir(dataset_path)
        subfolders = [f for f in all_items
                     if os.path.isdir(os.path.join(dataset_path, f))
                     and f.startswith('sca') and f.endswith('_data')]

        def extract_folder_num(folder_name):
            match = re.match(rf'{sca_prefix}(\d+)_data', folder_name)
            return int(match.group(1)) if match else 9999

        subfolders.sort(key=extract_folder_num)

        # 移除此打印，避免DDP重复打印

        for folder_name in subfolders:
            folder_path = os.path.join(dataset_path, folder_name)
            folder_num = extract_folder_num(folder_name)

            # 查找 edge 文件
            search_pattern = os.path.join(folder_path, f"edge_{sca_prefix}{folder_num}_*.txt")
            edge_files = glob.glob(search_pattern)

            file_ids = []
            pattern = re.compile(rf'edge_{sca_prefix}{folder_num}_(\d+)\.txt')

            for vf in edge_files:
                match = pattern.match(os.path.basename(vf))
                if match:
                    file_ids.append(int(match.group(1)))

            file_ids.sort()

            for data_id in file_ids:
                data_mapping[k_idx] = (folder_path, folder_num, str(data_id))
                k_idx += 1

            total_folders += 1

    n_total = k_idx - 1
    if is_main_process():
        print(f"初始化完成：总共扫描 {total_folders} 个文件夹，找到 {n_total} 组数据。")
    return data_mapping, n_total

def load_file_data(folder_path, prefix, folder_num, data_id):
    """通用数据读取函数"""
    filename = f"{prefix}_{sca_prefix}{folder_num}_{data_id}.txt"
    filepath = os.path.join(folder_path, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件缺失: {filepath}")
        
    data = np.loadtxt(filepath, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def build_graph_data(k):
    """
    构建图数据，包含残差计算步骤: r = b - A * Ebz
    """
    if not data_mapping:
        scan_all_data(root_data_path)
        
    if k not in data_mapping:
        raise ValueError(f"索引 k={k} 超出范围")
    
    folder_path, folder_num, data_id = data_mapping[k]
    
    # -------------------------------------------------------
    # 1. 基础数据读取
    # -------------------------------------------------------
    # 读取 edge (拓扑)
    edge_data = load_file_data(folder_path, "edge", folder_num, data_id)
    edge_index_np = edge_data.astype(np.int64) - 1 # MATLAB -> Python 索引
    edge_index = torch.tensor(edge_index_np.T, dtype=torch.long)
    num_nodes = int(edge_index.max()) + 1

    # 读取 eps (材料), b (右端项), Ebz (当前解/初始解)
    eps_data = load_file_data(folder_path, "eps", folder_num, data_id) # [N, 2]
    b_data = load_file_data(folder_path, "b", folder_num, data_id)     # [N, 2]
    Ebz_data = load_file_data(folder_path, "Ebz", folder_num, data_id) # [N, 2]
    
    # -------------------------------------------------------
    # 2. 计算残差 r (Preprocessing)
    # 公式: r = b - A * Ebz
    # -------------------------------------------------------
    
    # 2.1 读取稀疏矩阵 A 的信息 (Aij 和 Av)
    try:
        Aij_data = load_file_data(folder_path, "Aij", folder_num, data_id) # [NNZ, 2] 坐标
        Av_data = load_file_data(folder_path, "Av", folder_num, data_id)   # [NNZ, 2] 值
    except FileNotFoundError:
        raise FileNotFoundError(f"计算残差需要 Aij 和 Av 文件，但在数据组 {k} 中未找到。")

    # 2.2 构建 Scipy 稀疏矩阵 (COO 格式)
    # Aij 是 MATLAB 索引 (1-based)，需要减 1
    rows = Aij_data[:, 0].astype(int) - 1
    cols = Aij_data[:, 1].astype(int) - 1
    
    # 构造复数数值: Real + 1j * Imag
    values = Av_data[:, 0] + 1j * Av_data[:, 1]
    
    # 创建稀疏矩阵 A (N x N)
    A_mat = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
    
    # 2.3 准备向量 b 和 Ebz (复数形式)
    b_vec = b_data[:, 0] + 1j * b_data[:, 1]
    Ebz_vec = Ebz_data[:, 0] + 1j * Ebz_data[:, 1]
    
    # 2.4 执行矩阵乘法和减法: r = b - A * Ebz
    # A_mat.dot() 是高效的稀疏矩阵乘法
    Ax = A_mat.dot(Ebz_vec)
    r_vec = b_vec - Ax  # 得到复数残差向量 [N,]
    
    # 2.5 将残差拆分为实部和虚部 [N, 2]
    r_real = r_vec.real
    r_imag = r_vec.imag

    # -------------------------------------------------------
    # 3. 拼接节点特征 (Input Features)
    # 目标输入: [eps, r, Ebz] (共 6 通道)
    # -------------------------------------------------------
    # eps: [N, 2]
    # r:   [N, 2] (由上一步计算得到)
    # Ebz: [N, 2]
    
    x_tensor = torch.cat([
        torch.from_numpy(eps_data),                        # eps_real, eps_imag
        torch.tensor(r_real, dtype=torch.float32).unsqueeze(1), # r_real
        torch.tensor(r_imag, dtype=torch.float32).unsqueeze(1), # r_imag
        torch.from_numpy(Ebz_data),                             # Ebz_real, Ebz_imag (当前解)
        torch.from_numpy(Ebz_data),                             # bg_real, bg_imag (背景场，不随网络更新)
    ], dim=1)
    
    # -------------------------------------------------------
    # 4. 读取标签 (Ground Truth)
    # -------------------------------------------------------
    Esz_data = load_file_data(folder_path, "Esz", folder_num, data_id)
    y_tensor = torch.from_numpy(Esz_data)

    # -------------------------------------------------------
    # 5. 返回 Data 对象
    # -------------------------------------------------------
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    data.k_idx = torch.tensor([k])
    
    return data

# 测试运行
if __name__ == "__main__":
    scan_all_data(root_data_path)
    if n_total > 0:
        print("\n--- 读取并计算残差中 ---")
        try:
            data = build_graph_data(1)
            print("数据构建成功!")
            print(f"节点特征 x shape: {data.x.shape}")
            print("特征顺序: [eps_re, eps_im, r_re, r_im, Ebz_re, Ebz_im, bg_re, bg_im]")
            print(f"残差(r)部分均值: {data.x[:, 2:4].abs().mean().item():.4e}")
        except Exception as e:
            print(f"出错: {e}")