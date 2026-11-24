import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMStructureLearner(nn.Module):
    def __init__(self, dim_entity, dim_mole, hidden_dim=64, num_heads=4, dropout=0.2):
        """
        LLM 引导的结构学习模块
        
        Args:
            dim_entity: 实体 (诊断/手术) 的特征维度 (e.g., BioBERT embedding dim)
            dim_mole: 分子 (药物) 的特征维度 (e.g., MolT5 embedding dim)
            hidden_dim: 隐藏层维度
            num_heads: 多头注意力的头数
        """
        super(LLMStructureLearner, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # 线性投影，将不同模态的特征映射到同一语义空间
        self.project_entity = nn.Linear(dim_entity, hidden_dim * num_heads)
        self.project_mole = nn.Linear(dim_mole, hidden_dim * num_heads)
        
        # [New] Self-Attention 层，用于捕捉并发症上下文 (Context-Awareness)
        # 这使得每个 entity 的 embedding 能感知到同一次 visit 中的其他 entity
        self.context_encoder = nn.MultiheadAttention(
            embed_dim=dim_entity, 
            num_heads=4, 
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(dim_entity)
        
        # 门控机制，用于控制稀疏度 (LLM 的知识应该是稀疏的，不是所有药都能治所有病)
        self.gating = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 最后的图结构微调层
        # 初始时稍微偏向正值，保证梯度流动
        self.bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, entity_emb, mole_emb, static_graph=None):
        """
        Args:
            entity_emb: [num_entities, dim_entity] - 诊断或手术的语义向量
            mole_emb: [num_moles, dim_mole] - 药物的 MolT5 向量
            static_graph: [num_entities, num_moles] - 可选，原始的静态共现图作为先验
            
        Returns:
            dynamic_adj: [num_entities, num_moles] - 学习到的动态邻接矩阵 (权重)
        """
        # [New] 0. 上下文增强 (Context Encoding)
        # 输入是 [N, Dim]，我们需要把它变成 [1, N, Dim] 喂给 MultiheadAttention
        # entity_emb 代表当前 visit 的所有 entities，它们之间需要交互
        entity_seq = entity_emb.unsqueeze(0) # [1, N, Dim]
        
        # Self-Attention: Diag 之间互相看
        context_aware_emb, _ = self.context_encoder(entity_seq, entity_seq, entity_seq)
        
        # 残差连接 + LayerNorm
        entity_emb = self.layer_norm(entity_emb + context_aware_emb.squeeze(0))

        # 1. 投影到共同语义空间 [N, H*Heads]
        # entity_emb: (Num_Entities, Hidden)
        # mole_emb: (Num_Moles, Hidden)
        
        Q = self.project_entity(entity_emb) # Query: 病情 (Context-Aware)
        K = self.project_mole(mole_emb)     # Key: 药物机制
        
        # 2. 多头注意力计算相似度
        # Reshape to [N, Heads, Hidden]
        Q = Q.view(-1, self.num_heads, self.hidden_dim)
        K = K.view(-1, self.num_heads, self.hidden_dim)
        
        # Attention Score: [Num_Entities, Num_Moles, Heads]
        # Einstein Summation: n h d, m h d -> n m h
        attn_score = torch.einsum('nhd, mhd -> nmh', Q, K)
        
        # 3. 聚合多头信息 (Mean Pooling) -> [Num_Entities, Num_Moles]
        attn_score = attn_score.mean(dim=-1)
        
        # 4. 归一化与缩放
        attn_score = attn_score / (self.hidden_dim ** 0.5)
        
        # 5. 融合静态图先验 (Residual Connection)
        # 如果有静态图 (比如共现率)，我们希望 LLM 是去"修正"它，而不是完全重起炉灶
        if static_graph is not None:
            # 这里的 alpha 可以是一个可学习参数，或者固定值
            # 假设 static_graph 已经在 0-1 之间
            combined_score = attn_score + torch.log(static_graph + 1e-8) # Logits 融合
        else:
            combined_score = attn_score
            
        # 6. 生成最终的边权重 (0~1)
        # 使用 Sigmoid 激活，并加上 bias 偏置
        adj = torch.sigmoid(combined_score + self.bias)
        
        # 7. 稀疏化处理 (可选，但在医学图中很重要)
        # 只有权重超过一定阈值的边才被保留，或者用 Soft Thresholding
        adj = F.relu(adj - 0.01) # 简单的软阈值
        
        return adj
