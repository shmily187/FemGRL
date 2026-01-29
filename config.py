"""
PhiSAGE 项目全局配置文件
集中管理所有可配置的参数，避免循环导入问题
"""

# ==========================================
# 全局迭代次数配置（统一管理所有相关文件的迭代次数）
# ==========================================
N_ITER = 5  # 全局迭代次数，修改此值将影响所有相关文件

# ==========================================
# 全局模型保存路径配置
# ==========================================
SAVE_DIR = "/public/home/zzx/gnn/PhiSAGE/PhiSAGE-test/saved_models/C-final" # 模型保存目录
OUTPUT_DIR = "/public/home/zzx/gnn/PhiSAGE/PhiSAGE-test/training_outputs/C-final" # 训练曲线输出目录
# ==========================================
# 数据集配置
# ==========================================
# 数据集选择：设置为 'A' 或 'B'
# - 'A': 读取 data/A-4 目录（包含A-TainDataset等4个子目录），使用 scaA 命名格式
# - 'B': 读取 data/B-4 目录（包含B-TainDataset等4个子目录），使用 scaB 命名格式
# - 'C': 读取 data/C-4 目录（包含C-TainDataset等4个子目录），使用 scaC 命名格式
# 使用方法：
# 1. 将数据放在对应的目录结构中
# 2. 修改 DATASET_TYPE 的值
# 3. 程序会自动扫描所有子目录中的数据
#
DATASET_TYPE = 'A'  # 可选: 'A' 或 'B' 或 'C'
DATA_ROOT_PATH = "data/A-1"

# 根据选择动态设置相关参数（由 build_graph.py 使用）
if DATASET_TYPE == 'A':
    SCA_PREFIX = 'scaA'
    DATASET_DIRS_PATTERN = 'A-TainDataset*'
elif DATASET_TYPE == 'B':
    SCA_PREFIX = 'scaB'
    DATASET_DIRS_PATTERN = 'B-TainDataset*'
elif DATASET_TYPE == 'C':
    SCA_PREFIX = 'scaC'
    DATASET_DIRS_PATTERN = 'C-TainDataset*'
else:
    raise ValueError("DATASET_TYPE 必须设置为 'A' 或 'B' 或 'C'")
# ==========================================
# ==========================================
# 训练启动器配置 (run.py 使用)
# ==========================================
# 默认 GPU 配置
DEFAULT_TARGET_GPUS = "2,3"  # 默认使用的 GPU 列表

# 训练模式配置
DEFAULT_TRAIN_MODE = "ddp"  # 默认训练模式: "single", "multi", "ddp"

# 单卡训练配置
DEFAULT_SINGLE_GPU_ID = 0  # 默认单卡 GPU ID

# ==========================================
# DDP分布式训练配置
# ==========================================
# DDP通信端口配置
MASTER_PORT = "10970"  # DDP主进程通信端口

# ==========================================
# 网络维度配置
# ==========================================
# 网络架构配置：支持多种维度设置方式
#
# 方式1：使用base_dim自动计算（推荐）
# NETWORK_BASE_DIM = 56  # 基础维度，会自动生成 [56, 112, 192]
#
# 方式2：自定义维度配置
# NETWORK_CUSTOM_DIMS = [56, 112, 192]  # 手动指定每层维度
#
# 方式3：使用预设配置
# NETWORK_CONFIG = "default"  # 可选: "small", "medium", "large", "xlarge"

# 当前使用的配置方式
NETWORK_USE_CUSTOM_DIMS = True  # True=使用自定义维度，False=使用base_dim自动计算
NETWORK_BASE_DIM = 64            # 基础维度（当不使用自定义维度时）
NETWORK_CUSTOM_DIMS = [64, 128, 256]  # 自定义维度配置

# 池化配置
NETWORK_POOL_RATIOS = [0.8, 0.6]  # 池化比例

# ==========================================
# Loss函数配置
# ==========================================
# LOSS_TYPE: 选择使用的损失函数类型
# - "mse": 传统的MSE损失 (||pred - true||^2)
# - "phi": Phi损失 (||A*x - b||^2) - 直接计算物理残差
# - "asinh": Asinh损失 sqrt(asinh(norm(x-x_ref)^2/N)) - 平滑的损失函数
# - "hybrid": 混合损失 - 前100epoch纯MSE，后续MSE + λ*phi (λ从0.001开始每50epoch×10倍，至0.1)
LOSS_TYPE = "hybrid"  # 可选: "mse", "phi", "asinh" 或 "hybrid"

# ==========================================
# 预训练模型加载配置（迁移学习）
# ==========================================
# LOAD_PRETRAINED_MODEL: 是否加载预训练模型权重（仅模型权重，不加载优化器等状态）
# - True: 从 PRETRAINED_MODEL_DIR 目录加载模型权重，优化器等重新初始化
# - False: 不加载预训练模型，从头开始训练
LOAD_PRETRAINED_MODEL = False  # 是否加载预训练模型权重

# PRETRAINED_MODEL_DIR: 预训练模型目录
# 如果 LOAD_PRETRAINED_MODEL=True，将从该目录加载模型权重文件（real_iter_*.pth 和 imag_iter_*.pth）
# None表示使用SAVE_DIR，也可以指定其他目录
PRETRAINED_MODEL_DIR = None  # None表示使用SAVE_DIR，也可以指定其他目录

# ==========================================
# 其他全局配置参数
# ==========================================
# 可以在这里添加其他需要全局共享的配置参数