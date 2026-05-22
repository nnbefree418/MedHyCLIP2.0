import os  # 操作系统相关操作（本文件中未直接使用）
import math  # 数学运算模块
import torch  # PyTorch 主库
from torch import nn  # 神经网络模块
from torch.nn import functional as F  # 常用函数模块（本文件未直接使用）
from PIL import Image  # 图像读写（本文件未直接使用）
import geoopt  # 双曲几何库（用于 Hyper-MVFA）



# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    """
    残差式 CLIP 适配器模块（Adapter）
    作用：在不改动原始 CLIP 主干参数的前提下，引入轻量级的可训练层
        x --fc1--> 中间特征 --fc2--> 输出特征
    在主干上通常以残差方式叠加：x_out = 0.8*x + 0.1*seg_adapt_out + 0.1*det_adapt_out
    """

    def __init__(self, c_in, bottleneck=768):
        """
        :param c_in: 输入特征维度（例如 CLIP ViT-L 的通道数 1024）
        :param bottleneck: 中间瓶颈维度（默认 768）
        """
        super(ClipAdapter, self).__init__()
        # 第一个全连接 + 非线性，将特征降维到 bottleneck
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        # 第二个全连接 + 非线性，将特征升回原始维度 c_in
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        """
        :param x: 输入特征，形状 [N, L, C] 或 [L, N, C]，视调用位置而定
        :return: 
            x: 经过 fc1 后的中间特征（用于保存 patch tokens）
            y: 经过 fc2 后的输出特征（用于与主干特征融合）
        """
        x = self.fc1(x)  # 先通过瓶颈层降维
        y = self.fc2(x)  # 再升维回原通道数
        return x, y

        
class CLIP_Inplanted(nn.Module):
    """
    在预训练 CLIP 模型中“植入”适配器（Adapter）的封装模型：
    - 保留 CLIP 原有视觉编码器 image_encoder（ViT）
    - 在指定若干 transformer 层插入 seg_adapters（分割头）和 det_adapters（检测头）
    - 输出：
        pooled: 全局图像特征（CLIP 原生 pooled 输出）
        seg_patch_tokens: 若干层的分割 adapter 中间特征（用于像素级检测）
        det_patch_tokens: 若干层的检测 adapter 中间特征（用于图像/像素级检测）
    """

    def __init__(self, clip_model, features, use_hyperbolic=False, hyperbolic_c=0.1):
        """
        :param clip_model: 预训练 CLIP 模型（含 visual 编码器）
        :param features: 需要插入 Adapter 的层号列表（例如 [6, 12, 18, 24]，对应 transformer block 编号）
        :param use_hyperbolic: 是否使用双曲适配器（默认 False，保持原版 MVFA 行为）
        :param hyperbolic_c: 双曲空间曲率（默认 0.1）
        """
        super().__init__()
        self.clipmodel = clip_model  # 完整 CLIP 模型
        self.image_encoder = clip_model.visual  # 视觉编码器（ViT）
        self.features = features  # 需要插入 adapter 的层索引（1-based）
        self.use_hyperbolic = use_hyperbolic  # 双曲模式标志

        # 根据 use_hyperbolic 决定使用哪种 Adapter
        if self.use_hyperbolic:
            # 导入并使用双曲适配器
            from CLIP.hyperbolic_adapter import HyperbolicAdapter
            self.ball = geoopt.PoincareBall(c=hyperbolic_c)
            self.seg_adapters = nn.ModuleList(
                [HyperbolicAdapter(1024, bottle_dim=768, c=hyperbolic_c) for i in range(len(features))]
            )
            self.det_adapters = nn.ModuleList(
                [HyperbolicAdapter(1024, bottle_dim=768, c=hyperbolic_c) for i in range(len(features))]
            )
        else:
            # 使用原有欧氏适配器（保持向后兼容）
            self.seg_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])
            self.det_adapters = nn.ModuleList([ClipAdapter(1024, bottleneck=768) for i in range(len(features))])


    def forward(self, x):
        """
        :param x: 输入图像张量 [B, 3, H, W]
        :return:
            pooled: 全局 pooled 特征（与 CLIP 原始输出一致），形状 [B, C]
            seg_patch_tokens: 列表，每个元素为某一层的 seg adapter 中间特征 [B, L+1, C]
            det_patch_tokens: 列表，每个元素为某一层的 det adapter 中间特征 [B, L+1, C]
        """
        # 首先通过 ViT 的卷积打 patch（conv1 是 ViT 的 patch embedding）
        x = self.image_encoder.conv1(x)  # [B, C, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1)  # 展平空间维度，变为 [B, C, L]
        x = x.permute(0, 2, 1)  # 调整为 [B, L, C]

        # 拼接 CLS token：class_embedding + zeros 做 batch 扩展
        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # [B, L+1, C]
        # 加上位置编码
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        # Patch dropout（CLIP 内部的正则化操作）
        x = self.image_encoder.patch_dropout(x)
        # LN 预处理
        x = self.image_encoder.ln_pre(x)

        # ViT transformer 的输入为 [L+1, B, C]，因此需要再次 permute
        x = x.permute(1, 0, 2)  # [L+1, B, C]

        attn_out = []  # 用于存储指定层的注意力图（这里只在 i+1==12 时保存）
        seg_patch_tokens = []  # 存储分割 adapter 的中间特征
        det_patch_tokens = []  # 存储检测 adapter 的中间特征

        # 遍历 ViT 的 24 个 transformer block（以 i+1 对齐 CLIP 层编号）
        for i in range(24):
            # 特殊处理第 12 层：保存它的注意力图到 attn_out
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
                attn_out.append(attn)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            # 若当前层在指定的 features 列表中，则在该层插入 adapter
            if (i + 1) in self.features:
                adapter_idx = self.features.index(i + 1)
                
                if self.use_hyperbolic:
                    # ===== 双曲模式 =====
                    x_hyp = self.ball.expmap0(x)
                    seg_adapt_med, seg_adapt_out = self.seg_adapters[adapter_idx](x_hyp)
                    det_adapt_med, det_adapt_out = self.det_adapters[adapter_idx](x_hyp)
                    seg_adapt_out_eucl = self.ball.logmap0(seg_adapt_out)
                    det_adapt_out_eucl = self.ball.logmap0(det_adapt_out)
                    x = 0.8 * x + 0.1 * seg_adapt_out_eucl + 0.1 * det_adapt_out_eucl
                    seg_patch_tokens.append(seg_adapt_med)
                    det_patch_tokens.append(det_adapt_med)
                else:
                    # ===== 欧氏模式（原版 MVFA）=====
                    seg_adapt_med, seg_adapt_out = self.seg_adapters[adapter_idx](x)
                    det_adapt_med, det_adapt_out = self.det_adapters[adapter_idx](x)
                    x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
                    seg_patch_tokens.append(seg_adapt_med)
                    det_patch_tokens.append(det_adapt_med)

        # attn_out[0] 形状为 [B, num_heads, L, L]，这里 B, C, L 实际对应 [B, n_head, L]
        B, C, L = attn_out[0].shape
        # 去掉 CLS token 后，patch 数量为 L-1 = H*H，因此 H=sqrt(L-1)
        H = int(math.sqrt(L-1))
        # 初始化一个 H×H 的注意力图累积矩阵（直接放在 cuda 上）
        out_attn = torch.zeros([H, H]).to('cuda')

        # 将多个注意力头的注意力图叠加在 out_attn 上（这里只遍历 attn 的 head 维）
        #for i in range(len(attn)):
        for i in range(len(attn_out)):
            # 取第一个样本、第一个 head，从第 1 个 token（去掉 CLS）开始，reshape 为 [H, H]
            out_attn = out_attn + attn_out[i][0, 0, 1:].view(H, H)
        # Transformer 输出维度变回 [B, L+1, C]
        x = x.permute(1, 0, 2)

        # 将存储的 adapter 中间特征从 [L+1, B, C] 转为 [B, L+1, C]
        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        # 全局池化：CLIP 内部的 _global_pool，通常取 CLS token 或平均池化
        pooled, tokens = self.image_encoder._global_pool(x)
        # LN 后处理
        pooled = self.image_encoder.ln_post(pooled)

        # 若存在线性投影层（proj），则再做一次线性变换对齐文本空间
        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        # 返回：
        # pooled: 图像级全局特征
        # seg_patch_tokens: 分割适配器的中间 patch 特征（list，每层一个 [B, L+1, C]）
        # det_patch_tokens: 检测适配器的中间 patch 特征
        return pooled, seg_patch_tokens, det_patch_tokens
