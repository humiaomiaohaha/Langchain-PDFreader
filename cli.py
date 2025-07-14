#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from pdf_reader import PDFReader
from llm_interface import LLMInterface
from langchain_community.llms import HuggingFacePipeline
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def main():
    parser = argparse.ArgumentParser(description="智能PDF阅读器 - 命令行版本")
    parser.add_argument("--pdf-folder", default="pdfs", help="PDF文件所在文件夹")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="嵌入模型名称或本地路径")
    parser.add_argument("--llm-model", default="rule_based", help="语言模型名称 (rule_based, local:路径, hf:模型名, openai)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="文档块大小")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="文档块重叠大小")
    parser.add_argument("--interactive", action="store_true", help="进入交互模式")
    
    args = parser.parse_args()
    
    print("📚 智能PDF阅读器 - 命令行版本")
    print("=" * 50)
    
    try:
        print("正在初始化PDF阅读器...")
        pdf_reader = PDFReader(
            pdf_folder=args.pdf_folder,
            embedding_model=args.embedding_model
        )
        
        print(f"正在初始化语言模型: {args.llm_model}")
        llm_interface = LLMInterface(model_name=args.llm_model)
        
        model_info = llm_interface.get_model_info()
        print(f"✅ 语言模型初始化完成: {model_info['model_type']}")
        
        pdf_folder = Path(args.pdf_folder)
        if not pdf_folder.exists():
            print(f"PDF文件夹 {args.pdf_folder} 不存在，正在创建...")
            pdf_folder.mkdir(parents=True, exist_ok=True)
            print(f"请将PDF文件放入 {args.pdf_folder} 文件夹中")
            return
        
        print("正在加载PDF文件...")
        documents = pdf_reader.load_pdfs()
        
        if not documents:
            print(f"在 {args.pdf_folder} 中没有找到PDF文件")
            print("请将PDF文件放入该文件夹中")
            return
        
        print(f"成功加载 {len(documents)} 个文档")
        
        print("正在分割文档...")
        split_docs = pdf_reader.split_documents(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        print(f"文档分割完成，共 {len(split_docs)} 个块")
        
        print("正在创建向量数据库...")
        vectorstore = pdf_reader.create_vectorstore(split_docs)
        
        print("正在创建问答链...")
        llm_interface.create_qa_chain(vectorstore)
        
        print("✅ 初始化完成！")
        
        if args.interactive:
            interactive_mode(llm_interface)
        else:
            question = input("\n请输入您的问题 (或按Ctrl+C退出): ")
            answer = llm_interface.ask_question(question)
            print(f"\n答案: {answer['answer']}")
            
            if answer['source_documents']:
                print("\n相关文档片段:")
                for i, doc in enumerate(answer['source_documents']):
                    print(f"\n文档 {i+1}: {doc['metadata'].get('source', 'Unknown')}")
                    print(f"页码: {doc['metadata'].get('page', 'Unknown')}")
                    print(f"内容: {doc['content'][:200]}...")
            
    
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

def interactive_mode(llm_interface):
    print("\n🎯 进入交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n🤔 请输入您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if question.lower() == 'help':
                print_help()
                continue
            
            if not question:
                continue
            
            print("🤖 正在思考...")
            result = llm_interface.ask_question(question)
            
            if result["error"]:
                print(f"❌ 错误: {result['error']}")
            else:
                print(f"\n💡 答案: {result['answer']}")
                
                if result["source_documents"]:
                    print(f"\n📚 相关文档片段 ({len(result['source_documents'])} 个):")
                    for i, doc in enumerate(result['source_documents']):
                        print(f"\n--- 文档 {i+1} ---")
                        print(f"来源: {doc['metadata'].get('source', 'Unknown')}")
                        print(f"页码: {doc['metadata'].get('page', 'Unknown')}")
                        print(f"内容: {doc['content'][:300]}...")
        
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 处理问题时出现错误: {e}")

def print_help():
    help_text = """
📖 帮助信息:

基本命令:
- 直接输入问题: 系统会基于PDF内容回答您的问题
- help: 显示此帮助信息
- quit/exit/q: 退出程序

使用技巧:
1. 问题要具体明确，这样能获得更准确的答案
2. 可以询问PDF中的具体内容、概念解释等
3. 系统会显示相关的文档片段作为参考

示例问题:
- "这个文档主要讲了什么？"
- "请解释一下机器学习的基本概念"
- "文档中提到了哪些技术？"
"""
    print(help_text)

if __name__ == "__main__":
    main() 