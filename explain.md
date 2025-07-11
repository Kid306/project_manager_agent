# LangGraph RAG Agent 项目模块功能介绍

## 1. 核心架构概述

本项目是一个基于LangGraph和LangChain的智能RAG（检索增强生成）系统，设计用于处理复杂的信息检索和生成任务。系统通过图状态流程管理复杂对话工作流，支持工具调用与动态路由，并结合PostgreSQL实现会话和记忆的持久化存储。

项目整体架构采用了有向图结构设计，通过节点（Nodes）和边（Edges）定义智能代理的处理流程，实现了一个完整的企业级解决方案，包含多种大模型接入方式、分层记忆架构和智能工作流程。

![架构图](graph.png)

## 2. 主要模块说明

### 2.1 核心执行模块

#### 2.1.1 `main.py`

主程序入口，负责初始化FastAPI服务、数据库连接池和状态图，并提供REST API接口：

- 实现了完整的API生命周期管理，通过异步上下文管理器`lifespan`处理服务启动和关闭
- 提供`/v1/chat/completions`端点，支持流式和非流式响应
- 包含统一的错误处理和回复格式化函数
- 集成线上跟踪功能，支持LangSmith平台记录执行细节

#### 2.1.2 `demoRagAgent.py`

系统核心业务逻辑和工作流定义，包含状态图和所有处理节点：

- `ToolConfig`类：管理工具及其路由配置，根据工具类型动态配置最佳处理路径
- `ParallelToolNode`类：重定义的工具节点，支持并行处理多个工具调用，提高系统效率
- 状态处理节点：
  - `agent`：分析用户输入并决定是否调用工具
  - `grade_documents`：评估检索文档与用户问题的相关性
  - `rewrite`：当检索结果不理想时重写用户查询
  - `generate`：生成最终回复
  - `deal_tasks`：处理任务分解结果
- 路由函数：
  - `route_after_tools`：根据工具执行结果决定下一步处理节点
  - `route_after_grade`：根据文档评分结果决定是生成回复还是重写查询
  - `route_after_deal_tasks`：根据任务处理结果确定后续流程

#### 2.1.3 `webUI.py`

基于Gradio构建的Web用户界面：

- 提供聊天交互界面，支持流式输出回复
- 实现用户认证系统（注册、登录功能）
- 支持会话管理（创建新会话、查看历史会话）
- 通过REST API与后端服务交互，支持用户ID和会话ID维护

### 2.2 数据处理模块

#### 2.2.1 `vectorSaveTest.py`

知识库向量化处理模块：

- 支持多种大模型计算文本嵌入（OpenAI、OneAPI、阿里通义千问、Ollama）
- 实现批量向量化处理以提高效率
- 封装ChromaDB向量数据库操作，提供文档添加和检索功能
- 支持中英文PDF文档的处理和分割

#### 2.2.2 `utils/pdfSplitTest_Ch.py` & `utils/pdfSplitTest_En.py`

PDF文档处理工具：

- 提供中文和英文PDF文档的解析功能
- 实现智能段落分割，支持自定义段落长度
- 处理特殊字符和格式，保证文本质量

### 2.3 工具和配置模块

#### 2.3.1 `utils/tools_config.py`

工具定义和配置：

- 定义检索工具，连接向量数据库
- 实现需求分解工具，支持将复杂需求拆解为可执行任务
- 提供实用工具如`multiply`等

#### 2.3.2 `utils/llms.py`

大语言模型接口封装：

- 支持多种大模型接入（GPT系列、国产大模型、本地部署模型）
- 实现模型初始化、错误处理和重试逻辑
- 提供聊天模型和嵌入模型的统一接口

#### 2.3.3 `utils/config.py`

系统统一配置：

- 管理提示词模板文件路径
- 配置向量数据库参数
- 设置日志和数据库连接参数
- 提供API服务配置选项

#### 2.3.4 `utils/domain.py`

领域模型定义：

- `DocumentRelevanceScore`：文档相关性评分模型
- `RequirementBreakdownResult`：需求分解结果模型
- `MessagesState`：会话状态模型，管理消息、评分和重写计数

#### 2.3.5 `utils/helper.py`

辅助功能模块：

- 提供提示词模板加载和缓存
- 封装常用操作函数

### 2.4 提示词模板

位于`prompts/`目录下的各种模板文件：

- `prompt_template_agent_new.txt`：代理决策提示词
- `prompt_template_grade.txt`：文档评分提示词
- `prompt_template_rewrite.txt`：查询重写提示词
- `prompt_template_generate.txt`：回复生成提示词
- `prompt_template_deal_tasks.txt`：任务处理提示词
- `task_agent_prompt_template_requirement_breakdown.txt`：需求分解提示词

## 3. 工作流程

系统工作流程如下：

1. **用户查询处理**：系统接收用户输入，通过`agent`节点进行意图分析
2. **工具调用**：根据分析结果决定是否调用工具，支持并行执行多个工具
3. **智能路由**：根据工具执行结果动态决定后续流程：
   - 检索类工具路由到`grade_documents`节点评估文档质量
   - 任务分解工具路由到`deal_tasks`节点处理任务列表
   - 其他工具直接路由到`generate`节点生成回复
4. **文档评分**：评估检索文档与用户问题的相关性，决定是生成回复还是重写查询
5. **查询重写**：当文档相关性不足时，重写用户查询以提高检索质量
6. **回复生成**：根据上下文和工具结果生成最终回复

## 4. 技术亮点

1. **并行工具调用**：使用线程池技术并行执行多个工具，提高响应速度
2. **智能动态路由**：根据上下文状态自动选择最优处理路径
3. **数据库持久化**：使用PostgreSQL实现会话状态和记忆的可靠存储
4. **多模型支持**：灵活对接各种大模型，满足不同场景需求
5. **完整用户系统**：支持用户注册、登录和会话管理
6. **流式输出**：提供流式响应，增强用户体验
7. **错误恢复机制**：多层错误处理和重试逻辑，提高系统健壮性
8. **可视化状态图**：支持将复杂工作流导出为直观图形

## 5. 部署与使用

系统支持多种部署方式：

1. **本地开发环境**：直接运行`main.py`和`webUI.py`
2. **Docker容器化**：提供`docker-compose.yml`配置文件
3. **API服务**：通过FastAPI提供RESTful接口
4. **Web界面**：使用Gradio构建的直观用户界面

## 6. 总结

本项目展示了如何使用LangGraph构建复杂、有状态的智能代理系统，通过图结构有效管理对话流程和状态。系统设计充分考虑了企业级应用的需求，包括高可用性、可扩展性和易维护性，是一个功能完备的智能客服解决方案。 