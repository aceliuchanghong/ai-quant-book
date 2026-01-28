# Background: Multi-Agent Framework Comparison

> Multi-agent systems are becoming the standard architecture for complex AI applications. This article compares the features and use cases of mainstream frameworks.

---

## 1. Framework Overview

| Framework | Origin | Positioning | Open Source |
|-----------|--------|-------------|-------------|
| **Shannon** | Kocoro Labs | Production-grade orchestration | Yes |
| **AutoGen** | Microsoft | Conversational multi-agent | Yes |
| **CrewAI** | Community | Role-playing collaboration | Yes |
| **LangGraph** | LangChain | Graph-structured workflows | Yes |
| **MetaGPT** | DeepWisdom | Software development | Yes |

---

## 2. Framework Details

### 2.1 Shannon

**Positioning**: Production-grade multi-agent orchestration platform

**Core Features**:
- Go + Rust + Python multi-language architecture
- Temporal workflow engine (deterministic replay)
- Built-in budget control and cost tracking
- WASI sandbox execution
- OPA policy engine

**Architecture**:
```
Orchestrator (Go)     → Task routing, budget control
Agent Core (Rust)     → Sandbox execution
LLM Service (Python)  → Model calls, tool execution
Temporal              → Workflow orchestration
```

**Use Cases**:
- Production environment deployment
- Cost control requirements
- Deterministic debugging needs
- Security isolation requirements

**Example**:
```bash
curl -X POST http://localhost:8080/api/v1/tasks \
  -d '{"query": "Analyze market data", "session_id": "quant-1"}'
```

---

### 2.2 AutoGen

**Positioning**: Microsoft's open-source conversational multi-agent framework

**Core Features**:
- Conversation-based Agent interaction
- Human-in-the-loop support
- Code execution capabilities
- Flexible Agent definitions

**Architecture**:
```
UserProxyAgent    → Represents user, can execute code
AssistantAgent    → LLM-powered assistant
GroupChat         → Multi-Agent conversation management
```

**Use Cases**:
- Research and prototyping
- Code generation tasks
- Human-AI collaboration needs
- Conversational workflows

**Example**:
```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})

user_proxy.initiate_chat(assistant, message="Analyze this dataset")
```

---

### 2.3 CrewAI

**Positioning**: Role-playing multi-agent collaboration framework

**Core Features**:
- Clear role definitions (Role, Goal, Backstory)
- Task-driven
- Tool integration
- Simple and easy to use

**Architecture**:
```
Agent    → Define roles and capabilities
Task     → Define specific tasks
Crew     → Organize Agent collaboration
Tool     → Available tools for Agents
```

**Use Cases**:
- Rapid prototyping
- Tasks with clear roles
- Content generation
- Research assistants

**Example**:
```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role='Market Researcher',
    goal='Analyze market trends',
    backstory='You are a senior quantitative analyst'
)

task = Task(
    description='Analyze BTC trends over the past week',
    agent=researcher
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

---

### 2.4 LangGraph

**Positioning**: Graph-structured Agent workflow framework

**Core Features**:
- State machine + graph structure
- Loop and branch support
- LangChain ecosystem integration
- Workflow visualization

**Architecture**:
```
StateGraph   → Define states and transitions
Node         → Processing nodes (functions or Agents)
Edge         → State transition logic
Checkpointer → State persistence
```

**Use Cases**:
- Complex workflows
- Loop logic requirements
- State management needs
- LangChain users

**Example**:
```python
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_market)
workflow.add_node("decide", make_decision)
workflow.add_edge("analyze", "decide")

app = workflow.compile()
result = app.invoke({"input": "Analyze market"})
```

---

### 2.5 MetaGPT

**Positioning**: Multi-agent system for software development

**Core Features**:
- Simulates software teams
- Roles: PM, Architect, Engineer, QA
- Document-driven development
- Code generation

**Architecture**:
```
ProductManager → Requirements analysis
Architect      → System design
Engineer       → Code implementation
QA             → Testing and validation
```

**Use Cases**:
- Software development tasks
- Code generation
- Technical documentation generation

---

## 3. Comparison Matrix

| Feature | Shannon | AutoGen | CrewAI | LangGraph | MetaGPT |
|---------|---------|---------|--------|-----------|---------|
| Production Ready | 5/5 | 3/5 | 2/5 | 3/5 | 2/5 |
| Ease of Use | 3/5 | 4/5 | 5/5 | 3/5 | 3/5 |
| Cost Control | 5/5 | 2/5 | 2/5 | 2/5 | 2/5 |
| Debugging | 5/5 | 3/5 | 2/5 | 4/5 | 2/5 |
| Security Isolation | 5/5 | 2/5 | 1/5 | 2/5 | 2/5 |
| Community Activity | 3/5 | 5/5 | 4/5 | 4/5 | 3/5 |
| Quant Suitability | 4/5 | 3/5 | 2/5 | 3/5 | 1/5 |

---

## 4. Quant Trading Scenario Fit

### 4.1 Scenario: Market Analysis Multi-Agent

| Framework | Implementation Difficulty | Recommendation |
|-----------|---------------------------|----------------|
| Shannon | Medium | 5/5 |
| AutoGen | Low | 4/5 |
| CrewAI | Low | 3/5 |
| LangGraph | Medium | 4/5 |

### 4.2 Scenario: Real-Time Trade Execution

| Framework | Implementation Difficulty | Recommendation |
|-----------|---------------------------|----------------|
| Shannon | Medium | 5/5 |
| AutoGen | High | 2/5 |
| CrewAI | High | 1/5 |
| LangGraph | Medium | 3/5 |

**Reasons**:
- Shannon has production-grade orchestration and security sandbox
- AutoGen/CrewAI lack execution control
- Real-time trading requires low latency and high reliability

### 4.3 Scenario: Strategy Research Prototyping

| Framework | Implementation Difficulty | Recommendation |
|-----------|---------------------------|----------------|
| Shannon | Medium | 3/5 |
| AutoGen | Low | 5/5 |
| CrewAI | Low | 4/5 |
| LangGraph | Medium | 4/5 |

**Reason**: Research phase needs rapid iteration, ease of use matters more

---

## 5. Architecture Selection Decision Tree

```
What is your main requirement?
│
├─ Production Deployment
│   ├─ Need cost control → Shannon
│   ├─ Need deterministic replay → Shannon
│   └─ Need security sandbox → Shannon
│
├─ Rapid Prototyping
│   ├─ Conversational interaction → AutoGen
│   ├─ Role-playing tasks → CrewAI
│   └─ Complex workflows → LangGraph
│
├─ Code Generation Tasks
│   ├─ Need human-AI collaboration → AutoGen
│   └─ Simulate dev team → MetaGPT
│
└─ Quant Trading
    ├─ Research phase → AutoGen / CrewAI
    └─ Live trading → Shannon
```

---

## 6. Integration Examples

### 6.1 Shannon + Quant Strategy

```python
from shannon import ShannonClient

client = ShannonClient(base_url="http://localhost:8080")

# Submit analysis task
handle = client.submit_task(
    query="Analyze BTC/USDT market state, determine if trending or ranging",
    session_id="regime-detection"
)

# Wait for result
result = client.wait(handle.task_id)
```

### 6.2 AutoGen + Quant Research

```python
from autogen import AssistantAgent, UserProxyAgent

quant_analyst = AssistantAgent(
    "quant_analyst",
    system_message="You are a quantitative analyst skilled in market data analysis and strategy design",
    llm_config=llm_config
)

user = UserProxyAgent(
    "user",
    code_execution_config={"work_dir": "research"}
)

user.initiate_chat(
    quant_analyst,
    message="Design an RSI-based mean reversion strategy"
)
```

### 6.3 CrewAI + Research Team

```python
from crewai import Agent, Task, Crew

analyst = Agent(
    role='Quant Analyst',
    goal='Analyze market data',
    tools=[market_data_tool]
)

developer = Agent(
    role='Strategy Developer',
    goal='Convert analysis into code',
    tools=[code_execution_tool]
)

reviewer = Agent(
    role='Risk Control Reviewer',
    goal='Check strategy risks',
    tools=[risk_analysis_tool]
)

crew = Crew(
    agents=[analyst, developer, reviewer],
    tasks=[...],
    process="sequential"
)
```

---

## 7. Future Trends

1. **Standardization**: Agent communication protocol standards
2. **Specialization**: Domain-specific vertical frameworks
3. **Production-Ready**: More frameworks supporting production deployment
4. **Integration**: Multi-framework integration becomes common
5. **Low-Code**: Visual Agent orchestration tools

---

## 8. Selection Recommendations

| Scenario | Recommended Framework | Reason |
|----------|----------------------|--------|
| Live quant trading | Shannon | Production-grade, cost control, security |
| Quant strategy research | AutoGen | Easy to use, code execution |
| Rapid prototyping | CrewAI | Simplest, clear roles |
| Complex workflows | LangGraph | Graph structure, state management |
| Code generation | AutoGen / MetaGPT | Specifically optimized |

---

> **Core Principle**: There's no perfect framework, only the right choice for your scenario. Use simple frameworks for rapid validation in research, use robust frameworks for reliable operation in production.
