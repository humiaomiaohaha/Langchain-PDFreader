import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFReader:
    def __init__(self, pdf_folder: str = "pdfs", embedding_model: str = "D:\\HM\\Embedding"):
        self.pdf_folder = Path(pdf_folder)
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vectorstore = None
        self.documents = []

        self.pdf_folder.mkdir(exist_ok=True)

        self._init_embeddings()

    def _init_embeddings(self):
        try:
            logger.info(f"正在加载嵌入模型: {self.embedding_model}")

            self.model = SentenceTransformer(self.embedding_model)
            logger.info("嵌入模型加载成功")

        except Exception as e:
            logger.error(f"嵌入模型加载失败: {e}")
            raise

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            raise

    def load_pdfs(self) -> List[Document]:
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"在 {self.pdf_folder} 中没有找到PDF文件")
            return []

        documents = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"正在加载PDF: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["file_path"] = str(pdf_file)

                documents.extend(docs)
                logger.info(f"成功加载 {len(docs)} 页从 {pdf_file.name}")

            except Exception as e:
                logger.error(f"加载PDF {pdf_file.name} 失败: {e}")
                continue

        self.documents = documents
        logger.info(f"总共加载了 {len(documents)} 个文档")
        return documents

    def split_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        if not self.documents:
            logger.warning("没有文档可以分割")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        split_docs = text_splitter.split_documents(self.documents)
        logger.info(f"文档分割完成，共 {len(split_docs)} 个块")
        return split_docs

    def create_vectorstore(self, documents: List[Document], save_path: str = "vectorstore") -> FAISS:
        if not documents:
            logger.warning("没有文档可以创建向量数据库")
            return None

        try:
            logger.info("正在创建向量数据库...")

            texts = [doc.page_content for doc in documents]

            embeddings = self._get_embeddings(texts)

            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self._get_embeddings,
                metadatas=[doc.metadata for doc in documents]
            )

            self.vectorstore.save_local(save_path)
            logger.info(f"向量数据库已保存到: {save_path}")

            return self.vectorstore

        except Exception as e:
            logger.error(f"创建向量数据库失败: {e}")
            raise

    def load_vectorstore(self, save_path: str = "vectorstore") -> FAISS:
        try:
            if os.path.exists(save_path):
                logger.info(f"正在加载向量数据库: {save_path}")
                self.vectorstore = FAISS.load_local(save_path, self._get_embeddings)
                logger.info("向量数据库加载成功")
                return self.vectorstore
            else:
                logger.warning(f"向量数据库不存在: {save_path}")
                return None
        except Exception as e:
            logger.error(f"加载向量数据库失败: {e}")
            return None

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            logger.warning("向量数据库未初始化")
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_vectorstore_info(self) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"status": "未初始化"}

        try:
            return {
                "status": "已初始化",
                "index_size": self.vectorstore.index.ntotal,
                "dimension": self.vectorstore.index.d
            }
        except Exception as e:
            return {"status": f"获取信息失败: {e}"} 