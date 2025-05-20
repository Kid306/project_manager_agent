import json
from typing import Annotated, List

from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel

from .config import Config
from .domain import RequirementBreakdownResult, MessagesState
from .helper import create_chain
from .logger import logger

# 定义任务列表模型
class TaskList(BaseModel):
    tasks: List[RequirementBreakdownResult]

def get_tools(llm_embedding, task_llm):
    """
    创建并返回工具列表。

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储。
        task_llm: 用作任务分解的大模型

    Returns:
        list: 工具列表。
    """
    # 创建Chroma向量存储实例
    vectorstore = Chroma(
        persist_directory=Config.CHROMADB_DIRECTORY,
        collection_name=Config.CHROMADB_COLLECTION_NAME,
        embedding_function=llm_embedding,
    )
    # 将向量存储转换为检索器
    retriever = vectorstore.as_retriever()
    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description="这是员工信息查询工具。搜索并返回有关员工的所有相关档案信息。"
    )


    # 定义 multiply 工具
    @tool
    def multiply(a: float, b: float) -> float:
        """这是计算两个数的乘积的工具。返回最终的计算结果"""
        return a * b


    @tool
    def requirement_breakdown(
        requirement: str,
        state: Annotated[MessagesState, InjectedState],
    ) -> str:
        """
        这是需求分解工具，调用该工具来将需求分解成一个个可执行、可分配的子任务

        Args:
            requirement: 用户需求描述
            state: 当前状态

        Returns:
            str: 任务列表的JSON字符串，格式为：
            [
                {
                    "task_id": "1",
                    "description": "任务描述",
                    "dependencies": [],
                    "priority": 1,
                    "scope": "scope",
                    "duration": 1
                },
                ...
            ]
        """
        max_retries = 3
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                # 创建代理处理链
                agent_chain = create_chain(task_llm, Config.TASK_AGENT_PROMPT_TEMPLATE_TXT_REQUIREMENT_BREAKDOWN)
                
                # 构建提示，如果是重试，则添加错误信息
                prompt_input = {"requirement": requirement}
                if retry_count > 0 and last_error:
                    prompt_input["requirement"] = f"""
请重新尝试任务分解。上次尝试出现了以下错误：
{last_error}

请确保返回符合以下格式的任务列表：
{{{{
  [
    {{{{
      "task_id": "1",
      "description": "任务描述",
      "dependencies": [],
      "priority": 1,
      "scope": "scope",
      "duration": 1
    }}}},
    ...
  ]
}}}}

原始需求：{requirement}
"""
                # 调用代理链处理消息
                response = agent_chain.invoke(prompt_input)

                # 将任务列表存储到状态中
                state["task_breakdown_messages"] = response
                
                # 返回任务列表的JSON字符串或字符串表示
                if isinstance(response, str):
                    return response
                else:
                    return str(response)

            except Exception as e:
                retry_count += 1
                last_error = str(e)
                logger.error(f"Error in requirement_breakdown (attempt {retry_count}/{max_retries}): {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                if retry_count >= max_retries:
                    # 达到最大重试次数，返回错误信息
                    return f"Error after {max_retries} attempts: {last_error}"
                
                # 继续下一次重试
                continue

    # 返回工具列表
    # return [retriever_tool, multiply, requirement_breakdown]
    return [requirement_breakdown, retriever_tool]