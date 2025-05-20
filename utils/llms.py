import os
from typing import Tuple

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import logging


# Author:@南哥AGI研习社 (B站 or YouTube 搜索“南哥AGI研习社”)


# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 模型配置字典
MODEL_CONFIGS = {
    "openai": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "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",
        "chat_model": "openai/gpt-4o-mini",
        "embedding_base_url": "https://nangeai.top/v1",
        "embedding_api_key": "sk-iR65HZwPsfaepYlopSEN2btclqYahqhpr0oSXjHUE4rZzvv5",
        "embedding_model": "text-embedding-3-small"
    },
    # "oneapi": {
    #     "base_url": "http://139.224.72.218:3000/v1",
    #     "api_key": "sk-GseYmJ8pX1D0I00W7a506e8fDf23474A3C4B724FfD66aD9",
    #     "chat_model": "qwen-max",
    #     "embedding_model": "text-embedding-v1"
    # },
    # "qwen": {
    #     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    #     "api_key": "sk-c45db4d2628e48cf232326e152c9a537f",
    #     "chat_model": "qwen-max",
    #     "embedding_model": "text-embedding-v1"
    # },
    # "ollama": {
    #     "base_url": "http://localhost:11434/v1",
    #     "api_key": "ollama",
    #     "chat_model": "llama3.1:8b",
    #     "embedding_model": "nomic-embed-text:latest"
    # }
}


# 默认配置
DEFAULT_LLM_TYPE = "openai"
DEFAULT_TEMPERATURE = 0.7


class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    """
    初始化LLM实例

    Args:
        llm_type (str): LLM类型，可选值为 'openai', 'oneapi', 'qwen', 'ollama'

    Returns:
        ChatOpenAI: 初始化后的LLM实例

    Raises:
        LLMInitializationError: 当LLM初始化失败时抛出
    """
    try:
        # 检查llm_type是否有效
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # 特殊处理ollama类型
        if llm_type == "ollama":
            os.environ["OPENAI_API_KEY"] = "NA"

        # 创建LLM实例
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,  # 添加超时配置（秒）
            max_retries=2  # 添加重试次数
        )

        llm_embedding = OpenAIEmbeddings(
            base_url=config["embedding_base_url"],
            api_key=config["embedding_api_key"],
            model=config["embedding_model"],
            deployment=config["embedding_model"]
        )

        logger.info(f"成功初始化 {llm_type} LLM")
        return llm_chat, llm_embedding

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[
    ChatOpenAI, OpenAIEmbeddings]:
    """
    获取LLM实例的封装函数，提供默认值和错误处理

    Args:
        llm_type (str): LLM类型

    Returns:
        ChatOpenAI: LLM实例
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise  # 如果默认配置也失败，则抛出异常


def get_task_agent_model(llm_type: str = DEFAULT_LLM_TYPE) -> ChatOpenAI:
    return get_chat_model(llm_type)


# kid
def get_chat_model(llm_type: str = DEFAULT_LLM_TYPE) -> ChatOpenAI:
    """
    获取LLM实例的封装函数，提供默认值和错误处理

    Args:
        llm_type (str): LLM类型

    Returns:
        ChatOpenAI: LLM实例
    """
    try:
        chat_model = init_chat_model(llm_type)
        return chat_model
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            chat_model = init_chat_model(DEFAULT_LLM_TYPE)
            return chat_model
        raise  # 如果默认配置也失败，则抛出异常


def get_embedding_model(llm_type: str = DEFAULT_LLM_TYPE) -> OpenAIEmbeddings:
    """
获取LLM实例的封装函数，提供默认值和错误处理

Args:
    llm_type (str): LLM类型

Returns:
    ChatOpenAI: LLM实例
"""
    try:
        embedding_model = init_embedding_model(llm_type)
        return embedding_model
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            embedding_model = init_embedding_model(DEFAULT_LLM_TYPE)
            return embedding_model
        raise  # 如果默认配置也失败，则抛出异常


# kid
def init_embedding_model(llm_type: str = DEFAULT_LLM_TYPE):
    try:
        # 检查llm_type是否有效
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的Embedding Model类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # 特殊处理ollama类型
        if llm_type == "ollama":
            os.environ["OPENAI_API_KEY"] = "NA"

        llm_embedding = OpenAIEmbeddings(
            base_url=config["embedding_base_url"],
            api_key=config["embedding_api_key"],
            model=config["embedding_model"],
            deployment=config["embedding_model"]
        )

        logger.info(f"成功初始化 {llm_type} Embedding Model")
        return llm_embedding

    except ValueError as ve:
        logger.error(f"Embedding Model配置错误: {str(ve)}")
        raise LLMInitializationError(f"Embedding Model配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化Embedding Model失败: {str(e)}")
        raise LLMInitializationError(f"初始化Embedding Model失败: {str(e)}")

# kid
def init_chat_model(llm_type: str = DEFAULT_LLM_TYPE):
    try:
        # 检查llm_type是否有效
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # 特殊处理ollama类型
        if llm_type == "ollama":
            os.environ["OPENAI_API_KEY"] = "NA"

        # 创建LLM实例
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,  # 添加超时配置（秒）
            max_retries=2  # 添加重试次数
        )

        logger.info(f"成功初始化 {llm_type} LLM")
        return llm_chat

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")

# 示例使用
if __name__ == "__main__":
    try:
        # 测试不同类型的LLM初始化
        llm_openai = get_llm("openai")
        llm_qwen = get_llm("qwen")

        # 测试无效类型
        llm_invalid = get_llm("invalid_type")
    except LLMInitializationError as e:
        logger.error(f"程序终止: {str(e)}")