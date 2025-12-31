import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

# 从.env 文件中读取配置项并加载到系统环境中
load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 实例化递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter()
# 将长文档分割成适合模型处理的小块
chunks = text_splitter.split_documents(docs)

# 创建一个中文文本嵌入模型实例，用于将中文文本转换为数值向量，以便在向量数据库中进行相似性搜索
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 使用指定的嵌入模型创建内存向量存储
vectorstore = InMemoryVectorStore(embeddings)
# 将分割后的文档块添加到向量存储中
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)
