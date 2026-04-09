# Multi-Agent Systems Course Documentation

## Overview
Comprehensive documentation for the multi-agent systems course. Each document explains core concepts, patterns, and provides detailed code examples.

## Documentation Index

### Foundation
- [00-config.md](./00-config.md) - Configuration and setup
- [01-agent-loop.md](./01-agent-loop.md) - Bare-metal ReAct agent
- [01-langchain-agent.md](./01-langchain-agent.md) - LangChain agent API

### Core Patterns
- [02-structured-output.md](./02-structured-output.md) - JSON, Pydantic, structured output
- [03-tools-deep-dive.md](./03-tools-deep-dive.md) - Tool design patterns
- [04-memory-systems.md](./04-memory-systems.md) - Memory architectures
- [05-architectures.md](./05-architectures.md) - Plan-execute, reflexion, self-consistency

### Advanced Patterns
- [06-langgraph-fundamentals.md](./06-langgraph-fundamentals.md) - State graphs, checkpointing
- [07-multiagent-coordination.md](./07-multiagent-coordination.md) - Pipelines, fan-out, messaging
- [08-14-combined.md](./08-14-combined.md) - Days 8-14 combined (supervisor, RAG, code, evaluation, production, advanced, capstone)

## Quick Navigation by Topic

### Agent Basics
- Day 1: Basic agent loops
- Day 2: Structured output
- Day 3: Tool use

### Memory & State
- Day 4: Memory systems
- Day 6: LangGraph state management

### Architectures
- Day 5: Agent architectures
- Day 7: Multi-agent coordination
- Day 8: Supervisor pattern

### Production
- Day 9: RAG agents
- Day 10: Code agents
- Day 11: Evaluation
- Day 12: Production patterns

### Advanced
- Day 13: Advanced reasoning patterns
- Day 14: Capstone project

## How to Use These Docs

1. **Start with Day 1**: Understand basic agent loops
2. **Progress sequentially**: Each day builds on previous concepts
3. **Reference as needed**: Use docs while implementing your own agents
4. **Study patterns**: Focus on architectural patterns for your use case

## Key Concepts Reference

### ReAct Pattern
Reason → Act → Observe → Repeat

### Structured Output
- JSON output
- Pydantic models
- `.with_structured_output()`

### LangGraph Components
- **State**: Typed dictionary
- **Nodes**: Functions that transform state
- **Edges**: Connections between nodes
- **Conditional edges**: Routing based on state
- **Checkpointer**: Persistence

### Multi-Agent Patterns
- **Pipeline**: Sequential agents
- **Fan-out**: Parallel agents
- **Supervisor**: Central coordinator
- **Message passing**: Structured communication

## Dependencies
All scripts use:
- `langchain` - Core framework
- `langchain-ollama` - Ollama integration
- `langgraph` - State machines
- `pydantic` - Data validation
- `tenacity` - Retry logic

## Ollama Configuration
```python
OLLAMA_BASE_URL = "http://10.10.1.7:11434"
MODEL = "qwen3.5:9b"
```

## Running Scripts
```bash
# Run a single script
python day01/agent_loop.py

# Run all scripts (custom runner)
python run_all_fixed.py
```

## Contributing
When adding new patterns:
1. Create script in appropriate day folder
2. Add documentation to this docs folder
3. Update this README with new links

## License
Same as course repository.
