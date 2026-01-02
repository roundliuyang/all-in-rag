import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

# 1. 初始化设置
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://172.26.118.30:19530"


# 2. 定义工具 (编码器和可视化函数)
class Encoder:
    """编码器类，用于将图像和文本编码为向量。"""

    def __init__(self, model_name: str, model_path: str):
        # 初始化Visualized_BGE模型
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        # 设置模型为评估模式（关闭dropout等）
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        """
        对图像和文本进行联合编码
        Args:
            image_path: 图像文件路径
            text: 查询文本
        Returns:
            编码后的向量列表
        """
        with torch.no_grad():  # 禁用梯度计算，节省内存
            query_emb = self.model.encode(image=image_path, text=text)  # 执行编码
        return query_emb.tolist()[0]  # 转换为列表并返回第一个元素

    def encode_image(self, image_path: str) -> list[float]:
        """
        对单张图像进行编码
        Args:
            image_path: 图像文件路径
        Returns:
                编码后的向量列表
        """
        with torch.no_grad():  # 禁用梯度计算
            query_emb = self.model.encode(image=image_path)  # 只对图像编码
        return query_emb.tolist()[0]  # 转换为列表并返回第一个元素


def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300, img_width: int = 300,
                      row_count: int = 3) -> np.ndarray:
    """
    从检索到的图像列表创建一个全景图用于可视化
    Args:
        query_image_path: 查询图像路径
        retrieved_images: 检索到的图像路径列表
        img_height: 每张图像的高度
        img_width: 每张图像的宽度
        row_count: 每行显示的图像数量
    Returns:
        合并后的全景图像
    """
    # 计算全景图尺寸
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    # 创建白色背景的全景图像
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    # 创建查询图像显示区域
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    # 添加红色边框
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    # 将带边框的查询图像放置在显示区域底部
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    # 添加"Query"标签
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count   # 计算当前图像在网格中的位置
        start_row, start_col = row * img_height, col * img_width     # 计算在全景图中的起始坐标

        retrieved_pil = Image.open(img_path).convert("RGB")      # 打开图像并转换为RGB
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]       # 转换为OpenCV格式(BGR)
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))   # 调整大小并预留边框空间
        # 添加黑色边框
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # 将处理后的图像放置到全景图的对应位置
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved

        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)

    # 水平拼接查询图像区域和结果图像区域
    return np.hstack([query_display_area, panoramic_image])


# 3. 初始化客户端
print("--> 正在初始化编码器和Milvus客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 4. 创建 Milvus Collection
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
# 检查是否存在同名集合，如果有则删除
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

# 获取dragon目录下所有png图像
image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")
# 获取向量维度（通过编码一张图像获取）
dim = len(encoder.encode_image(image_list[0]))

# 定义集合的字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),    # 主键ID字段，自动递增
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),               # 向量字段，维度为dim
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),         # 图像路径字段
]

# 创建集合 Schema
schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构:")
print(schema)

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))

# 5. 准备并插入数据
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []
# 遍历图像列表，为每张图像生成向量并准备插入数据
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path)   # 编码图像
    data_to_insert.append({"vector": vector, "image_path": image_path})    # 添加到插入列表

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")

# 6. 创建索引
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
# 准备索引参数
index_params = milvus_client.prepare_index_params()
# 添加HNSW索引（适用于高效相似性搜索）
index_params.add_index(
    field_name="vector",    # 索引字段
    index_type="HNSW",      # 索引类型
    metric_type="COSINE",   # 相似度度量方式
    params={"M": 16, "efConstruction": 256}   # HNSW参数
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")
print("索引详情:")
print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))
# 加载集合到内存以供搜索
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")

# 7. 执行多模态检索
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
# 设置查询图像和查询文本
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"
# 对查询图像和文本进行联合编码
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

# 执行向量搜索
search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],     # 搜索向量
    output_fields=["image_path"],  # 返回字段
    limit=5,   # 限制返回结果数量
    search_params={"metric_type": "COSINE", "params": {"ef": 128}}   # 搜索参数
)[0]

retrieved_images = []
print("检索结果:")
# 遍历搜索结果并显示详细信息
for i, hit in enumerate(search_results):
    print(f"  Top {i + 1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])    # 收集检索到的图像路径

# 8. 可视化与清理
print(f"\n--> 正在可视化结果并清理资源")
if not retrieved_images:
    print("没有检索到任何图像。")
else:
    # 生成可视化结果
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    # 保存结果图像
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到: {combined_image_path}")
    Image.open(combined_image_path).show()

# 释放集合资源
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
# 删除集合
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")
