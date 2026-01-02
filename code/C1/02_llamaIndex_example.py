import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 从指定路径加载文档数据
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 基于加载的文档创建向量存储索引
index = VectorStoreIndex.from_documents(docs)

# 从索引创建查询引擎，用于处理用户查询
query_engine = index.as_query_engine()

# 打印查询引擎使用的提示词模板
print(query_engine.get_prompts())

# 向文档内容提问并打印结果
print(query_engine.query("文中举了哪些例子?"))