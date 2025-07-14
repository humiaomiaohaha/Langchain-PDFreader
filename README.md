# PDF阅读器项目Bug记录与解决方案

## 项目概述
基于LangChain的智能PDF阅读器，支持多种语言模型后端（规则系统、本地Transformers、HuggingFace、OpenAI API）。


## LangChain兼容性问题

### Bug : Chain.__call__ 方法弃用警告
**错误信息：**
```
LangChainDeprecationWarning: The method `Chain.__call__` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
```

**原因：** LangChain版本更新，`__call__`方法被弃用。

**解决方案：**
```python
# 修改前
result = self.qa_chain({"query": question})

# 修改后
result = self.qa_chain.invoke({"query": question})
```

### Bug : Pydantic版本兼容性警告
**错误信息：**
```
LangChainDeprecationWarning: As of langchain-core 0.3.0, LangChain uses pydantic v2 internally. The langchain_core.pydantic_v1 module was a compatibility shim for pydantic v1, and should no longer be used.
```

**原因：** LangChain升级到pydantic v2，但有一些组件还使用v1。

**解决方案：**
```python
# 更新导入
from pydantic import BaseModel  # 使用v2
# 而不是
from langchain_core.pydantic_v1 import BaseModel  # 已弃用
```

---

## 模型加载与初始化问题

### Bug : 缺少accelerate包
**错误信息：**
```
Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`. You can install it with `pip install accelerate`
```

**原因：** 使用`device_map="auto"`需要accelerate包支持。

**解决方案：**
```bash
pip install accelerate
```

### Bug : 模型下载网络连接失败
**错误信息：**
```
ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。'))
```

**原因：** 网络连接不稳定。

**解决方案：**

# 从本地缓存加载



### Bug : 模型需要trust_remote_code参数
**错误信息：**
```
The repository THUDM/chatglm2-6b contains custom code which must be executed to correctly load the model. Please pass the argument `trust_remote_code=True`
```

**原因：** 好像有一些模型包含自定义代码，需要显式信任。

**解决方案：**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True  # 添加此参数
)
```



---

## Prompt工程问题

### Bug : Prompt过长导致模型无法处理
**错误信息：**
```
Token indices sequence length is longer than the specified maximum sequence length for this model (1840 > 1024). Running this sequence through the model will result in indexing errors
Input length of input_ids is 1840, but `max_length` is set to 512.
```

**原因：** RetrievalQA把所有文档都拼成大prompt了，超出小模型的处理能力。

**解决方案：**
```python
# 1. 减少检索文档数量
retriever=vectorstore.as_retriever(search_kwargs={"k": 1})

# 2. 限制prompt长度
max_prompt_length = 500
if len(prompt) > max_prompt_length:
    prompt = prompt[:max_prompt_length] + "..."

# 3. 检查输入token长度
if inputs.shape[1] > 800:
    inputs = inputs[:, -800:]

```

### Bug : Prompt格式不适合小模型
**问题描述：** 复杂的prompt模板导致小模型瞎说。

**原因：** 小模型（如GPT-2）无法理解复杂的指令格式。

**解决方案：**
```python
# 简化prompt模板


```

### Bug : 中文Prompt与英文模型不兼容
**问题描述：** 中文prompt输入英文模型（如GPT-2）导致输出乱码或无意义内容。

**原因：** GPT-2是英文模型，无法理解中文prompt。

**解决方案：**
```python
# 1. 换成中文模型

# 2. 或使用API模型
model_name = "openai"  # 支持多语言
```

### Bug : Prompt中的特殊字符导致tokenization错误
**错误信息：**
```
Tokenization error: unexpected character
```

**原因：** Prompt中包含特殊字符或格式。

**解决方案：**
```python
# 移除prompt中的特殊字符

```

---

## 网络连接问题

### Bug : HuggingFace Hub缓存警告
**错误信息：**
```
UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\...\.cache\huggingface\hub\models--gpt2.
```

**原因：** Windows系统不支持symlinks，导致缓存效率降低。

**解决方案：**
```

# 在代码中设置
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
```

### Bug : 模型下载超时
**错误信息：**
```
Connection timeout after 30 seconds
```

**原因：** 网络连接慢或模型文件过大。

**解决方案：**
```python
# 
#设置代理
```

---

## 自定义LLM接口问题

### Bug : 自定义LLM不是Runnable实例
**错误信息：**
```
2 validation errors for LLMChain
llm.is-instance[Runnable]
Input should be an instance of Runnable
```

**原因：** 自定义LLM类没有正确实现LangChain的Runnable接口。

**解决方案：**
```python
# 使用LangChain官方的HuggingFacePipeline
```

### Bug : 自定义LLM返回空字符串
**问题描述：** 自定义LLM能独立工作，但在LangChain链中返回空字符串。

**原因：** LangChain期望特定的返回格式，自定义LLM格式不匹配。

**解决方案：**
```python
# 确保返回正确的格式
#改成
def invoke(self, input_data, config=None):
    if isinstance(input_data, dict):
        prompt = input_data.get("text", str(input_data))
    else:
        prompt = str(input_data)
    
    response = self.__call__(prompt)
    return {"text": response}  # 返回字典格式
```

---

## 模型输出质量问题


### Bug : 模型输出被截断
**问题描述：** 模型回答不完整，被强制截断。

**原因：** `max_tokens`设置过小。

**解决方案：**
```python
# 增加`max_tokens
```

### Bug : 模型输出包含输入内容
**问题描述：** 输出包含完整的输入prompt，而不是只返回新生成的内容。

**原因：** 没有正确移除输入部分。

**解决方案：**
```python
# 移除输入部分
response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
if response.startswith(prompt):
    response = response[len(prompt):].strip()
```
