from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
import json
import re
import torch
import requests
import streamlit as st
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from transformers import RagRetriever, RagSequenceForGeneration, RagTokenizer

# 加载知识库
with open('kanboard_methods.json', 'r') as f:
    knowledge_base = json.load(f)

# # 初始化 RAG 模型
# tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')
# retriever = RagRetriever.from_pretrained('facebook/rag-token-nq', index_name='exact', passages_path='kanboard_methods.json')
# modelrag = RagSequenceForGeneration.from_pretrained('facebook/rag-token-nq', retriever=retriever)

# 初始化 DPR 模型
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

def TaskSplitter(text):
    # 定义提示模板
    prompt = PromptTemplate(
        input_variables=["task_description"],
        template=(
            "Please split the following task into logically ordered subtasks, ensuring each step is dependent on the successful completion of prior steps."
        "For example, if a project needs to be checked before adding tasks, first verify its existence or create it if needed, retrieve any relevant project information, and then proceed to subsequent steps."
        "For each subtask, indicate if it can be executed using the Kanboard API "
        "(use 'Yes' if executable, and 'No' otherwise). "
        "There are 4 default column in a project,Backlog,Ready,Work in Progress and Done"
        "Return the output in JSON format as a list of dictionaries, where each dictionary contains: "
        "'subtask' (the subtask description) and 'can_use_api' (whether it can be done via API). "
        "Output format example: "
        "[{{'subtask': 'Describe task', 'can_use_api': 'Yes'}}, {{'subtask': 'Discuss project', 'can_use_api': 'No'}}]. "
        "Task: {task_description}"
        )
    )
    
    # 将提示与模型组合
    chain = prompt | llm
    # 调用链并获取响应
    response = chain.invoke({"task_description": text})
    
    # 返回模型的文本输出
    return response

def ParameterIdentifier(subtasks_json):
    # 定义提示模板
    prompt = PromptTemplate(
        input_variables=["subtasks"],
        template=(
            "For each of the following subtasks that are marked as executable using the Kanboard API, "
        "identify the type of task (project, column, or task) and extract the necessary parameters. "
        "Fill in the specified parameters based on the task type, setting any parameter not mentioned to an empty value. "
        "The fields to populate for each type are as follows:\n"
        "- Project: 'project_id', 'name', 'description'\n"
        "- Column: 'project_id', 'column_id'\n"
        "- Task: 'project_id', 'column_id', 'title', 'due_date' (use the format YYYY-MM-DD)\n"
        "For project related subtask,name should be the project name,for task related subtask,title should be the task name"
        "For column related subtask,there are four default columns in a project,they are: "
        "'Blacklog' with column_id = 1, 'Ready'with column_id = 2, 'Work in Progress' with column_id = 3 and 'Done' with column_id = 4 "
        "Return the output in JSON format as a dictionary, where each key is the subtask description and the value is "
        "a dictionary of the extracted parameters.\n"
        "Subtasks: {subtasks}"
        )
    )
    
    # 将提示与模型组合
    chain = prompt | llm
    # 调用链并获取响应
    response = chain.invoke({"subtasks": subtasks_json})
    
    # 返回模型的文本输出
    return response

# def MethodSelector(subtasks_params_json):

#     if hasattr(subtasks_params_json, 'content'):
#         subtasks_params_json = subtasks_params_json.content

#     subtasks_params_json = re.sub(r'```json|\n```', '', subtasks_params_json)  

#     subtasks_params = json.loads(subtasks_params_json)
#     selected_methods = {}
    
#     for subtask, params in subtasks_params.items():
#         # 构建查询字符串
#         query = f"Find the most appropriate method for the subtask: {subtask} with parameters: {params}"
        
#         # 编码查询
#         inputs = tokenizer([query], return_tensors='pt')
        
#         # 获取检索结果
#         retrieved_docs = retriever.retrieve(inputs['input_ids'])
        
#         # 使用 RAG 模型生成答案
#         generated_answers = modelrag.generate(inputs['input_ids'], context=retrieved_docs)
        
#         # 解码生成的答案
#         answers = tokenizer.batch_decode(generated_answers, skip_special_tokens=True)
        
#         # 提取方法名和参数
#         for answer in answers:
#             for method, info in knowledge_base.items():
#                 if info['description'] in answer:
#                     selected_methods[subtask] = {
#                         'method': method,
#                         'params': {param: params.get(param, '') for param in info['parameters']}
#                     }
#                     break
#             if subtask in selected_methods:
#                 break
    
#     return json.dumps(selected_methods)

def MethodSelector(subtasks_params_json):
    # 如果 subtasks_params_json 是一个对象，尝试提取其 content 属性
    if hasattr(subtasks_params_json, 'content'):
        subtasks_params_json = subtasks_params_json.content

    # 移除 Markdown 格式的标记，确保只剩下 JSON 内容
    subtasks_params_json = re.sub(r'```json|\n```', '', subtasks_params_json)

    # 确保 subtasks_params_json 是一个字符串
    if not isinstance(subtasks_params_json, str):
        raise ValueError("subtasks_params_json must be a string")
    
    st.write(f"Type of subtasks_params_json after extraction: {type(subtasks_params_json)}")

    # 输出调试信息，查看 subtasks_params_json 的内容
    st.write(f"Content of subtasks_params_json: {subtasks_params_json}")

    # 尝试解析 JSON 字符串
    try:
        subtasks_params = json.loads(subtasks_params_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    selected_methods = {}
    
    for subtask, params in subtasks_params.items():
        # 构建查询字符串
        query = f"Find the most appropriate method for the subtask: {subtask} with parameters: {params}"
        
        # 编码查询
        question_inputs = tokenizer(query, return_tensors='pt')
        question_embedding = question_encoder(**question_inputs).pooler_output
        
        # 编码上下文
        context_embeddings = []
        contexts = []
        for method, info in knowledge_base.items():
            context = f"{info['description']} with parameters: {info['parameters']}"
            context_inputs = tokenizer(context, return_tensors='pt')
            context_embedding = context_encoder(**context_inputs).pooler_output
            context_embeddings.append(context_embedding)
            contexts.append(context)
        
        # 计算相似度并选择最匹配的方法
        scores = torch.cosine_similarity(question_embedding, torch.stack(context_embeddings), dim=-1)
        best_match_index = torch.argmax(scores)
        best_match_context = contexts[best_match_index]
        
        # 提取方法名和参数
        for method, info in knowledge_base.items():
            if info['description'] in best_match_context:
                selected_methods[subtask] = {
                    'method': method,
                    'params': {param: params.get(param, '') for param in info['parameters']}
                }
                break
    
    return json.dumps(selected_methods)

class APITools(BaseTool):
    name: str = "APITools"
    description: str = "Executes the chosen Kanboard API method with parameters"

    def _run(self, method: str, params: dict):
        # 构建请求负载
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": 1,
            "params": params
        }

        # 发送 POST 请求到 Kanboard API
        response = requests.post(
            KANBOARD_API_URL,
            auth=('jsonrpc', API_TOKEN),
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )

        # 处理响应并返回结果
        if response.status_code == 200:
            result = response.json()
            if 'result' in result:
                return result['result']
            else:
                return result.get('error', 'Unknown error')
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    async def _arun(self, *args, **kwargs):
        # 异步方法（如果需要）
        raise NotImplementedError("APITools does not support async")

# 获取 Kanboard API 的 URL 和令牌
KANBOARD_API_URL='http://localhost:8181/jsonrpc.php'
API_TOKEN ='4dd50c005ce8df513d7ad790a122a7c9cd410bad94fc573ca34be594d7ce'

# 设置 OpenAI 的 API 基础 URL 和密钥
base_url = "https://apix.ai-gaochao.cn/v1"
api_key = "sk-x0Storo8fEfKLPHFC229A67aDb4d427d9320D44cF56d4a74"  # 请替换为您的实际 API 密钥

# 初始化 ChatOpenAI 模型
llm = ChatOpenAI(
    model='gpt-4o',  # 确保模型名称正确
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url
)

tools = [
    StructuredTool.from_function(
        func=TaskSplitter,
        name="TaskSplitter",
        description="Splits user input into subtasks"
    ),
    StructuredTool.from_function(
        func=ParameterIdentifier,
        name="ParameterIdentifier",
        description="Identifies parameters for Kanboard API calls when subtasks are executable by API"
    ),
    StructuredTool.from_function(
        func=MethodSelector,
        name="MethodSelector",
        description="Selects appropriate API method and parameters after parameters identified"
    ),
    APITools()  # 实例化 APITools
]



# 修改后的 prompt
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""Analyze and execute the input request step-by-step, STRICTLY using ONLY the following tools:
- `TaskSplitter`
- `ParameterIdentifier`
- `MethodSelector`
- `APITools`
When passing `action_input`, make sure for Tool `APITools`, it is in JSON format.When you need information you don't know, use 'getBoard' to get all information in that specify project.

Input: {input}

Steps:
1. **Task Splitting**: Use `TaskSplitter` to divide the input task into subtasks. Identify each subtask and determine whether it requires a Kanboard API call. Store the result as a structured list of subtasks.

2. **Parameter Identification**: For each identified and API-executable subtask, use `ParameterIdentifier` to extract the necessary parameters for the API call.

3. **Method Selection**: Pass each subtask's parameters to `MethodSelector` to choose the appropriate API method and prepare the required arguments for calling the Kanboard API.

4. **API Execution**: Using `APITools`, execute each chosen API method with the prepared parameters. Update any missing or dependent parameters dynamically based on previous steps (e.g., retrieve and fill in project or column IDs if needed).

Final Output:
For each subtask, list:
   - **Subtask**: [Subtask description]
   - **Method**: [Selected API method]
   - **Parameters**: [Parameters used in API call]
   - **Result**: [API response]

{agent_scratchpad}
"""
)


# 使用定义好的 tools 和 prompt 创建 agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    prompt=prompt,
    verbose=True,
    max_iterations=20,  # 设置合理的最大迭代次数
    early_stopping_method="force"  # 在达到最大迭代次数后强制停止
)


# Streamlit 部分
st.title("iKB Task Management_RAG")

# 输入框
user_input = st.text_input("请输入任务描述:")

# 按钮

if st.button("提交"):
    try:
        # 调试 TaskSplitter 步骤
        task_split_output = TaskSplitter(user_input)
        st.write("Task Split Output:", task_split_output)
        
        # 调试 ParameterIdentifier 步骤
        subtask_output = ParameterIdentifier(task_split_output)
        st.write("Parameter Identification Output:", subtask_output)
        
        # 调试 MethodSelector 步骤
        methods_output = MethodSelector(subtask_output)
        st.write("Method Selection Output:", methods_output)
        
        # 最终调用代理执行器
        result = agent({"input": user_input})
        st.write("Agent Execution Result:", json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        st.write("发生错误:", str(e))

