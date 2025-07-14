import logging
from typing import List, Dict, Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import json

logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, model_name: str = "rule_based"):
        self.model_name = model_name
        self.llm = None
        self.qa_chain = None
        
        self._init_llm()
    
    def _init_llm(self):
        try:
            logger.info(f"正在初始化语言模型: {self.model_name}")
            
            if self.model_name.startswith("local:"):
                self._init_local_transformers()
            elif self.model_name in ["openai", "gpt-3.5-turbo", "gpt-4"]:
                self._init_openai_api()
            elif self.model_name.startswith("hf:"):
                self._init_huggingface_model()
            else:
                self._init_rule_based()
                
        except Exception as e:
            logger.error(f"语言模型初始化失败: {e}")
            print(f"❌ 语言模型初始化失败: {e}")
            print("🔄 回退到规则系统")
            self._init_rule_based()
    
    def _init_local_transformers(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_path = self.model_name.replace("local:", "")
            logger.info(f"正在加载本地模型: {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            class LocalLLM:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                
                def __call__(self, prompt: str) -> str:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=512,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return response.replace(prompt, "").strip()
            
            self.llm = LocalLLM(model, tokenizer)
            logger.info("本地Transformers模型初始化成功")
            
        except Exception as e:
            logger.error(f"本地Transformers模型初始化失败: {e}")
            raise
    
    def _init_openai_api(self):
        try:
            from langchain_community.llms import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("未设置OPENAI_API_KEY环境变量")
            
            self.llm = OpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=api_key
            )
            logger.info("OpenAI API语言模型初始化成功")
            
        except Exception as e:
            logger.error(f"OpenAI API初始化失败: {e}")
            raise
    
    def _init_huggingface_model(self):
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from transformers.pipelines import pipeline
            
            model_name = self.model_name.replace("hf:", "")
            logger.info(f"正在加载HuggingFace模型: {model_name}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
                print(f"✅ 从本地缓存加载模型: {model_name}")
            except Exception as e:
                print(f"本地缓存加载失败，尝试在线下载: {e}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("HuggingFace模型初始化成功")
            
        except Exception as e:
            logger.error(f"HuggingFace模型初始化失败: {e}")
            raise
    
    def _init_rule_based(self):
        logger.info("使用规则系统作为语言模型")
        
        try:
            from langchain_community.llms import FakeListLLM
            
            responses = [
                "根据文档内容，我可以为您提供相关信息。",
                "基于PDF文档的分析，相关内容如下：",
                "根据检索到的文档片段，答案如下：",
                "基于文档内容，我找到了以下相关信息：",
                "根据PDF文档，相关内容是："
            ]
            
            self.llm = FakeListLLM(responses=responses)
            logger.info("规则系统初始化成功")
            
        except Exception as e:
            logger.error(f"规则系统初始化失败: {e}")
            class SimpleLLM:
                def __call__(self, prompt: str) -> str:
                    return f"基于文档内容，{prompt}的相关信息如下：\n\n[这里会显示从PDF中检索到的相关内容]"
            
            self.llm = SimpleLLM()
    
    def create_qa_chain(self, vectorstore):
        try:
            if not self.llm:
                raise ValueError("语言模型未初始化")
            
            template = """{context}

Q: {question}
A:"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
                chain_type_kwargs={"prompt": prompt}
            )
            
            logger.info("问答链创建成功")
            
        except Exception as e:
            logger.error(f"创建问答链失败: {e}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        try:
            if not self.qa_chain:
                raise ValueError("问答链未初始化，请先调用create_qa_chain")
            
            result = self.qa_chain.invoke({"query": question})
            
            answer = result.get("result", "")
            if not answer or answer.strip() == "":
                answer = "模型没有生成有效回答，请尝试重新提问或使用不同的模型。"
            
            return {
                "answer": answer,
                "source_documents": self._format_source_documents(result.get("source_documents", [])),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"提问失败: {e}")
            return {
                "answer": f"抱歉，处理您的问题时出现错误: {e}",
                "source_documents": [],
                "error": str(e)
            }
    
    def _format_source_documents(self, docs) -> List[Dict[str, Any]]:
        formatted_docs = []
        for doc in docs:
            formatted_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return formatted_docs
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": type(self.llm).__name__,
            "is_available": self.llm is not None
        } 