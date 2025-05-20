from typing import TypedDict, Annotated, Sequence, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class DocumentRelevanceScore(BaseModel):
    # 定义binary_score字段，表示相关性评分，取值为"yes"或"no"
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class RequirementBreakdownResult(BaseModel):
    task_id: str = Field(description="任务的唯一id（从1开始递增）")
    description: str = Field(description="任务的详细描述，概括了该任务需要做的具体工作")
    dependencies: list[str] = Field(description="任务所依赖的前置任务id集合，表示了任务执行的先后顺序")
    priority: int = Field(description="当前任务的优先级，默认从1开始。数字越大优先级越高")
    scope: str = Field(description="任务所属的范畴。server表示后端、front表示前端、ux表示UED设计、pm表示产品经理、qa表示测试人员、other表示其他")
    duration: int = Field(description="该任务完成所需时长，单位为'天'")


# 定义消息状态类，使用TypedDict进行类型注解
class MessagesState(TypedDict):
    # 定义messages字段，类型为消息序列，使用add_messages处理追加
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 定义relevance_score字段，用于存储文档相关性评分
    relevance_score: Annotated[Optional[str], "Relevance score of retrieved documents, 'yes' or 'no'"]
    # 定义rewrite_count字段，用于跟踪问题重写的次数，达到次数退出graph的递归循环
    rewrite_count: Annotated[int, "Number of times query has been rewritten"]

    # 任务分解历史消息
    task_breakdown_messages: list[str]