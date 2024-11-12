from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
import os
from dotenv import load_dotenv
import json
import requests
import streamlit as st
#import plotly.graph_objects as go

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

def MethodSelector(subtasks_params_json):
    # 定义方法及其所需参数
    method_requirements = {
        "createProject": ["name"],
        "getProjectByName": ["name"],
        "getBoard": ["project_id"],
        "createTask": ["title", "project_id"],
        "closeTask": ["task_id"],
        "moveTaskPosition": ["project_id","task_id","column_id"]
        
    }
    
    # 定义提示模板
    prompt = PromptTemplate(
        input_variables=["subtasks_params", "method_requirements"],
        template=(
            "Given the subtasks and their parameters: {subtasks_params}, "
            "For each subtask, choose the most appropriate method from the following list of methods with required parameters: "
            "{method_requirements}. "
            "Output in JSON format as a dictionary where each key is the subtask and the value is another dictionary with "
            "'method' and 'params'.\n"
            "Example Output:\n"
            "{{\n"
            "  'Subtask description': {{'method': 'selected_method', 'params': {{'param1': 'value1', 'param2': 'value2'}}}},\n"
            "  ...\n"
            "}}"
        )
    )
    
    # 将提示与模型组合
    chain = prompt | llm
    # 调用链并获取响应
    response = chain.invoke({
        "subtasks_params": subtasks_params_json,
        "method_requirements": method_requirements
    })
    
    # 返回模型的文本输出
    return response



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

        

# 初始化工具
# 从 .env 文件加载环境变量
load_dotenv()

# 获取 Kanboard API 的 URL 和令牌
#KANBOARD_API_URL = os.getenv("KANBOARD_API_URL")
#API_TOKEN = os.getenv("API_TOKEN")
KANBOARD_API_URL='http://localhost:8081/jsonrpc.php'
API_TOKEN ='6b9df2687fe1d57a41f8f9bcd4fef44b9703c99187485d43a839826204d3'

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
st.title("iKB Task Management")

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
# 定义一个示例 Kanboard 项目数据处理函数，展示每列的任务
# def visualize_kanban_board(tasks):
#     # 假设有四个默认的看板列
#     columns = ["Backlog", "Ready", "Work in Progress", "Done"]
#     tasks_by_column = {col: [] for col in columns}
    
#     # 将任务分配到相应的列中
#     for task in tasks:
#         column_name = task.get("column", "Backlog")  # 获取任务的列名，默认 "Backlog"
#         task_name = task.get("task", "Unnamed Task")  # 获取任务名称
#         tasks_by_column[column_name].append(task_name)

#     # 创建看板图表
#     fig = go.Figure()
#     for col in columns:
#         fig.add_trace(go.Bar(
#             x=[col] * len(tasks_by_column[col]),
#             y=[1] * len(tasks_by_column[col]),
#             text=tasks_by_column[col],
#             textposition='auto',
#             orientation='h',
#             width=0.5
#         ))

#     # 设置图表布局
#     fig.update_layout(
#         title="Kanboard Task Visualization",
#         xaxis=dict(title="Columns"),
#         yaxis=dict(title="Tasks", showticklabels=False),
#         barmode='stack',
#         showlegend=False,
#         height=500
#     )

#     # 在 Streamlit 中显示图表
#     st.plotly_chart(fig)

# # Streamlit 主程序
# st.title("Kanboard Task Management Visualization")

# # 输入框
# user_input = st.text_input("请输入任务描述:")

# # 提交按钮
# if st.button("提交"):
#     try:
#         # 执行 TaskSplitter 步骤
#         task_split_output = TaskSplitter(user_input)
#         st.write("Task Split Output:", task_split_output)
        
#         # 执行 ParameterIdentifier 步骤
#         subtask_output = ParameterIdentifier(task_split_output)
#         st.write("Parameter Identification Output:", subtask_output)
        
#         # 执行 MethodSelector 步骤
#         methods_output = MethodSelector(subtask_output)
#         st.write("Method Selection Output:", methods_output)
        
#         # 最终调用代理执行器
#         result = agent({"input": user_input})
        
#         # 将 AIMessage 转换为字符串或 JSON 格式进行显示
#         try:
#             result_str = str(result)
#             st.write("Agent Execution Result:", result_str)
#         except Exception as e:
#             st.write("发生错误:", str(e))
        
#         # 将结果转换为适合看板可视化的格式
#         kanban_tasks = []
#         for subtask, details in json.loads(methods_output).items():
#             kanban_tasks.append({
#                 "column": details.get("method", "Backlog"),  # 使用 API 方法作为列名
#                 "task": subtask  # 使用子任务描述作为任务名称
#             })

#         # 可视化 Kanban 看板
#         visualize_kanban_board(kanban_tasks)

#     except Exception as e:
#         st.write("发生错误:", str(e))
