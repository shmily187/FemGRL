import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TopKPooling

# 从全局配置文件导入迭代次数配置
from config import N_ITER

# ==========================================
# 6. 网络维度配置导入
# ==========================================
# 从全局配置导入网络维度设置
from config import (
    NETWORK_USE_CUSTOM_DIMS,
    NETWORK_BASE_DIM,
    NETWORK_CUSTOM_DIMS,
    NETWORK_POOL_RATIOS
)

# ==========================================
# 0. BatchNorm 辅助函数
# ==========================================
def get_batch_norm(num_features):
    return nn.BatchNorm1d(num_features)

# ==========================================
# 1. 物理感知图卷积层
# ==========================================
class EMFullComplexLayer(MessagePassing):
    def __init__(self, in_feats, out_feats):
        super(EMFullComplexLayer, self).__init__()
        self.aggr = 'mean'
        total_in_dim = in_feats + in_feats
        self.lin_fusion = nn.Linear(total_in_dim, out_feats)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggs, x):
        combined = torch.cat([x, aggs], dim=1)
        fused_msg = self.lin_fusion(combined)
        return fused_msg

# ==========================================
# 2. 修改后的 GCN 模块
# ==========================================
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.conv = EMFullComplexLayer(in_feats, out_feats)
        self.bn = get_batch_norm(out_feats)
        self.gelu = nn.GELU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class FFTFeatureLayer(nn.Module):
    """
    对特征维度进行 FFT
    """
    def __init__(self, in_feats, out_feats):
        super(FFTFeatureLayer, self).__init__()
        # 线性投影调整维度
        self.lin_in = nn.Linear(in_feats, out_feats)
        
        # 可学习的频域滤波器 (复数权重)
        # RFFT 后，频域维度为 (out_feats // 2) + 1
        self.freq_dim = out_feats // 2 + 1
        
        # 初始化复数权重
        # shape: [freq_dim] - 对每个频率分量进行缩放
        self.complex_weight = nn.Parameter(
            torch.randn(self.freq_dim, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        # x: [N, in_feats] -> [N, out_feats]
        x = self.lin_in(x)

        # 修复：cuFFT在float16模式下只支持维度大小为2的幂次
        # 如果输入是float16且维度不是2的幂次，需要转换为float32
        original_dtype = x.dtype
        converted_to_float32 = False
        if x.dtype == torch.float16:
            # 检查最后一个维度是否为2的幂次
            last_dim = x.size(-1)
            is_power_of_two = (last_dim & (last_dim - 1)) == 0 and last_dim != 0
            if not is_power_of_two:
                # 转换为float32以避免cuFFT限制
                x = x.to(torch.float32)
                converted_to_float32 = True

        x_fft = torch.fft.rfft(x, dim=-1)
        
        # 2. 频域滤波 / 混合
        weight = torch.view_as_complex(self.complex_weight)
        # 广播乘法: [N, freq_dim] * [freq_dim]
        x_fft = x_fft * weight        
        # 3. 傅里叶逆变换 (Complex -> Real)
        x_out = torch.fft.irfft(x_fft, n=x.size(-1), dim=-1)
        
        # 如果原始是float16且被转换过，转换回float16（保持精度一致性）
        if original_dtype == torch.float16 and converted_to_float32:
            x_out = x_out.to(original_dtype)

        return x_out

# ==========================================
# 2.5 [核心修改] FFM 增强的输入层
# ==========================================
class FFMEncodingLayer(nn.Module):
    """
    使用傅里叶特征映射 (FFM) 替换传统的线性输入层。
    逻辑：
    1. 输入 features (如坐标或物理量)
    2. 映射 -> sin(2*pi*B*x), cos(2*pi*B*x)
    3. 线性投影融合
    4. 物理图卷积 (EMFullComplexLayer)
    """
    def __init__(self, in_feats, out_feats, sigma=1.0, mapping_size=None):
        super(FFMEncodingLayer, self).__init__()
        
        # 1. 配置 FFM 参数
        self.input_dim = in_feats
        # 如果未指定 mapping_size，默认设为 out_feats 的一半（因为sin/cos会翻倍）
        # 或者设为一个固定的高维空间，例如 128 或 256
        self.mapping_size = mapping_size if mapping_size is not None else max(out_feats, 64)
        self.sigma = sigma
        
        # 2. 初始化高斯随机矩阵 B (不可学习，类似位置编码)
        # shape: [in_feats, mapping_size]
        self.B = nn.Parameter(
            torch.randn(self.input_dim, self.mapping_size) * self.sigma, 
            requires_grad=False
        )
        
        # FFM 后的维度 = mapping_size * 2 (sin + cos)
        ffm_out_dim = self.mapping_size * 2
        
        # 3. 特征融合层：将高维 FFM 特征投影回目标维度
        self.feature_projection = nn.Linear(ffm_out_dim, out_feats)
        
        # 4. 空间混合：保留图卷积能力
        self.conv = EMFullComplexLayer(out_feats, out_feats)
        
        # 5. 激活与归一化
        self.bn = get_batch_norm(out_feats)
        self.gelu = nn.GELU()

    def input_mapping(self, x):
        # x: [N, input_dim]
        # 投影: (2 * pi * x) @ B
        # 结果 shape: [N, mapping_size]
        x_proj = (2.0 * torch.pi * x) @ self.B
        
        # 拼接 sin 和 cos -> [N, mapping_size * 2]
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

    def forward(self, x, edge_index):
        # 1. FFM 映射 (提升频率感知能力)
        x_ffm = self.input_mapping(x)
        
        # 2. 线性投影 (融合特征)
        x_embed = self.feature_projection(x_ffm)
        
        # 3. 图卷积 (聚合邻居信息)
        x_out = self.conv(x_embed, edge_index)
        
        # 4. 后处理
        x_out = self.bn(x_out)
        x_out = self.gelu(x_out)
        
        return x_out

# ==========================================
# 2.6 新增：混合谱图卷积层 (替代原 GCN)
# ==========================================
class SpectralGCN(nn.Module):
    """
    混合层：结合 局部图卷积(GCN) 和 频域特征变换(FFT)
    Out = GCN(x, edge) + FFT(x)
    """
    def __init__(self, in_feats, out_feats):
        super(SpectralGCN, self).__init__()
        
        # 分支1：物理感知图卷积 (处理局部连接)
        self.spatial_conv = EMFullComplexLayer(in_feats, out_feats)
        
        # 分支2：FFT 频域层 (处理全局/频域特征)
        self.spectral_conv = FFTFeatureLayer(in_feats, out_feats)
        
        # 融合门控 (可学习的加权系数)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.bn = get_batch_norm(out_feats)
        self.gelu = nn.GELU()

    def forward(self, x, edge_index):
        # 1. 空间路径：消息传递
        x_spatial = self.spatial_conv(x, edge_index)
        
        # 2. 频谱路径：FFT 变换 (不依赖 edge_index，纯节点特征变换)
        x_spectral = self.spectral_conv(x)
        
        # 3. 残差融合
        # 使用 sigmoid 确保融合比例在 0-1 之间，或者直接相加
        # 这里使用加权求和，兼顾空间和频域信息
        x_out = x_spatial + self.alpha * x_spectral
        
        # 4. 激活与归一化
        x_out = self.bn(x_out)
        x_out = self.gelu(x_out)
        
        return x_out

# ==========================================
# 2.7 FFM处理
# ==========================================
class ffm_process(nn.Module):
    def __init__(self, in_feats, out_feats, ffm_sigma=1.0):
        super(ffm_process, self).__init__()
        self.B = nn.Parameter(torch.randn(in_feats, out_feats) * ffm_sigma, requires_grad=False)

    def forward(self, x):
        x_proj = (2.0 * torch.pi * x) @ self.B
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

# ==========================================
# 3. Graph U-Nets 组件 (Checkpoint稳定版)
# ==========================================
class gPool(nn.Module):
    """Graph Pooling layer - 简化版，不特殊处理背景场"""
    def __init__(self, in_dim, ratio):
        super(gPool, self).__init__()
        self.ratio = ratio
        self.proj = nn.Linear(in_dim, 1)  # 评分投影
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        N, C = x.size()

        # 计算节点重要性得分（使用所有特征）
        scores = self.proj(x)  # [N, 1]
        scores = self.sigmoid(scores).squeeze()  # [N]

        # 确定保留的节点数量
        k = max(1, int(self.ratio * N))

        # 选择top-k节点
        values, idx = torch.topk(scores, k)

        # 门控机制：保留的节点特征按重要性加权
        x_pool = x[idx] * values.view(-1, 1)  # [k, C]

        # 筛选保留节点之间的边
        row, col = edge_index
        row_mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        col_mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        row_mask[idx] = True
        col_mask[idx] = True
        edge_mask = row_mask[row] & col_mask[col]
        edge_index_pool = edge_index[:, edge_mask]

        # 重新映射节点索引到0-k范围
        idx_map = -torch.ones(N, dtype=torch.long, device=x.device)
        idx_map[idx] = torch.arange(k, device=x.device)
        edge_index_pool[0] = idx_map[edge_index_pool[0]]
        edge_index_pool[1] = idx_map[edge_index_pool[1]]

        return x_pool, edge_index_pool, idx

class gUnpool(nn.Module):
    """Graph Unpooling layer (inverse of Pool)"""
    def __init__(self):
        super(gUnpool, self).__init__()
        
    def forward(self, x_pool, x_skip, idx):
        # idx 对应 TopKPooling 返回的 perm (保留节点的索引)
        N = x_skip.size(0)
        C = x_pool.size(1)

        # 恢复原始图结构大小
        x_unpool = torch.zeros(N, C, device=x_pool.device, dtype=x_skip.dtype)

        if idx is not None and len(idx) > 0:
            # 填充保留节点的特征
            # PyG TopKPooling 的 perm 保证了 idx < N
            # 确保 x_pool 的行数与 idx 长度一致
            if x_pool.size(0) == len(idx):
                x_unpool[idx] = x_pool
            else:
                # 理论上不应发生，除非维度对不上
                valid_len = min(x_pool.size(0), len(idx))
                x_unpool[idx[:valid_len]] = x_pool[:valid_len]

        # 跳跃连接（残差连接）
        x_unpool = x_unpool + x_skip

        return x_unpool

# ==========================================
# 4. 辅助模块
# ==========================================
class StackedLinearBlock(nn.Module):
    def __init__(self, in_feats, out_feats, dropout):
        super(StackedLinearBlock, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)
        self.bn = get_batch_norm(out_feats)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x

class FinalLinear(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(FinalLinear, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, a=1.0)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        return x

# ==========================================
# 5. BuildGCN 网络构建 (最接近原始版本)
# ==========================================
class BuildGCN(nn.Module):
    def __init__(self, input_feats, output_feats, base_dim=None, custom_dims=None):
        super(BuildGCN, self).__init__()

        # 优先级：函数参数 > 全局配置 > 默认值
        if custom_dims is not None:
            # 函数参数指定的自定义维度（最高优先级）
            dims = custom_dims
        elif base_dim is not None:
            # 函数参数指定的基础维度
            dims = [base_dim, base_dim * 2, base_dim * 4]
        elif NETWORK_USE_CUSTOM_DIMS:
            # 全局配置的自定义维度
            dims = NETWORK_CUSTOM_DIMS
        else:
            # 全局配置的基础维度自动计算
            dims = [NETWORK_BASE_DIM, NETWORK_BASE_DIM * 2, NETWORK_BASE_DIM * 4]

        # 使用全局配置的池化比例
        pool_ratios = NETWORK_POOL_RATIOS
        
        # ========== 编码器路径 ==========
        self.gcn1 = SpectralGCN(input_feats, dims[0])
        self.pool1 = TopKPooling(dims[0], pool_ratios[0])
        self.gcn2 = GCN(dims[0], dims[1])
        self.pool2 = TopKPooling(dims[1], pool_ratios[1])
        self.gcn3 = GCN(dims[1], dims[2])
        
        # ========== 瓶颈层 ==========
        self.bottom_gcn = GCN(dims[2], dims[2])
        
        # ========== 解码器路径 ==========
        self.dec_gcn1 = GCN(dims[2], dims[1])
        self.unpool1 = gUnpool()
        self.dec_gcn2 = GCN(dims[1], dims[0])
        self.unpool2 = gUnpool()
        self.dec_gcn3 = GCN(dims[0], dims[0])
        
        # ========== 输出层 ==========
        self.lin_stacked = StackedLinearBlock(dims[0], dims[0], dropout=0.5)
        self.final_linear = FinalLinear(dims[0], output_feats)
        
    def forward(self, ndata, edata, batch=None):
        x, edge_index = ndata, edata

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # === 编码器路径 ===
        x1 = self.gcn1(x, edge_index)

        # 第1次池化 (TopKPooling)
        # 返回值: x, edge_index, edge_attr, batch, perm, score
        x2, edge_index2, _, batch2, idx1, _ = self.pool1(x1, edge_index, batch=batch)
        
        x2 = self.gcn2(x2, edge_index2)

        # 第2次池化 (TopKPooling)
        x3, edge_index3, _, batch3, idx2, _ = self.pool2(x2, edge_index2, batch=batch2)
        
        x3 = self.gcn3(x3, edge_index3)

        # === 瓶颈层 ===
        x_bottom = self.bottom_gcn(x3, edge_index3)

        # === 解码器路径 ===
        # 注意：这里继续使用 edge_index3，和原逻辑一致
        x_up1 = self.dec_gcn1(x_bottom, edge_index3)
        # 反池化：需要传入 idx (即 perm) 来恢复到上一层的大小 (x2 的大小)
        x_up1 = self.unpool1(x_up1, x2, idx2)

        # 注意：这里使用 edge_index2
        x_up2 = self.dec_gcn2(x_up1, edge_index2)
        # 反池化：恢复到输入层大小 (x1 的大小)
        x_up2 = self.unpool2(x_up2, x1, idx1)

        x_out = self.dec_gcn3(x_up2, edge_index)

        # === 输出层 ===
        x_out = self.lin_stacked(x_out)
        x_out = self.final_linear(x_out)

        return x_out


# ==========================================
# 7. BuildGCNList (保持不变)
# ==========================================
class BuildGCNList(nn.Module):
    def __init__(self, input_feats, output_feats, n_iter=N_ITER, base_dim=None, custom_dims=None):
        super(BuildGCNList, self).__init__()
        self.n_iter = n_iter
        self.input_feats = input_feats
        self.output_feats = output_feats
        self.networks = nn.ModuleList()

        for i in range(n_iter):
            network = BuildGCN(input_feats, output_feats, base_dim=base_dim, custom_dims=custom_dims)
            self.networks.append(network)
            setattr(self, f'iter_{i}', network)
        
        print(f"已创建 {n_iter} 个集成Graph U-Nets的BuildGCN网络")
    
    def forward(self, ndata, edata, batch=None, iter_idx=None):
        if iter_idx is not None:
            if iter_idx < 0 or iter_idx >= self.n_iter:
                raise ValueError(f"iter_idx必须在[0, {self.n_iter-1}]范围内")
            return self.networks[iter_idx](ndata, edata, batch)
        else:
            outputs = []
            for i in range(self.n_iter):
                output = self.networks[i](ndata, edata, batch)
                outputs.append(output)
            return outputs
    
    def get_network(self, iter_idx):
        return self.networks[iter_idx]
    
    def __len__(self):
        return self.n_iter