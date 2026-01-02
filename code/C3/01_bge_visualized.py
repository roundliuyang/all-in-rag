import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

# 初始化可视化BGE模型，指定基础模型名称和权重路径
model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth")
# 设置模型为评估模式，关闭梯度计算
model.eval()

# 在无梯度计算模式下进行推理
with torch.no_grad():
    # 编码纯文本："datawhale开源组织的logo"
    text_emb = model.encode(text="datawhale开源组织的logo")
    # 编码第一张图片
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    # 编码第一张图片与文本的多模态信息
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale开源组织的logo")
    # 编码第二张图片
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    # 编码第二张图片与文本的多模态信息
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")

# 计算不同嵌入向量之间的相似度（通过点积计算）
sim_1 = img_emb_1 @ img_emb_2.T      # 纯图像1 vs 纯图像2 的相似度
sim_2 = img_emb_1 @ multi_emb_1.T    # 纯图像1 vs 图文结合1 的相似度
sim_3 = text_emb @ multi_emb_1.T     # 纯文本 vs 图文结合1 的相似度
sim_4 = multi_emb_1 @ multi_emb_2.T  # 图文结合1 vs 图文结合2 的相似度

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}")    # 输出多模态嵌入向量的维度
print(f"图像向量维度: {img_emb_1.shape}")       # 输出图像嵌入向量的维度
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")   # 显示多模态向量的前10个值
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")    # 显示图像向量的前10个值
