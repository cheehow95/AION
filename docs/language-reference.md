# AION Language Reference

## Table of Contents
1. [Overview](#overview)
2. [Declarations](#declarations)
3. [Agents](#agents)
4. [Memory Types](#memory-types)
5. [Models](#models)
6. [Tools](#tools)
7. [Policies](#policies)
8. [Reasoning Blocks](#reasoning-blocks)
9. [Control Flow](#control-flow)
10. [Actions](#actions)
11. [Expressions](#expressions)

---

## Overview

AION (Artificial Intelligence Oriented Notation) is a declarative, AI-native programming language designed to build **thinking systems** rather than algorithms.

```aion
agent Assistant {
  goal "Help users effectively"
  
  memory working
  model LLM
  
  on input(question):
    think
    analyze question
    decide answer
    respond
}
```

---

## Declarations

### Agent Declaration

```aion
agent AgentName {
  goal "Primary objective"
  # ... agent body
}
```

### Model Declaration

```aion
model ModelName {
  provider = "openai"    # openai, anthropic, ollama, local
  name = "gpt-4"
}
```

### Tool Declaration

```aion
tool tool_name {
  trust = medium    # low, medium, high
  cost = low        # low, medium, high
}
```

### Policy Declaration

```aion
policy {
  max_tokens = 2048
  allow_web = true
}
```

---

## Agents

An **agent** is an autonomous reasoning unit with goals, memory, and capabilities.

### Agent Body Members

| Member | Description |
|--------|-------------|
| `goal` | The agent's primary objective |
| `memory` | Memory stores for the agent |
| `model` | Reference to a model |
| `tool` | Reference to a tool |
| `policy` | Agent-specific policies |
| `on` | Event handlers |

### Example

```aion
agent Researcher {
  goal "Find accurate information"
  
  memory working
  memory long_term
  
  model LLM
  
  tool web_search
  
  on input(query):
    think
    use web_search(query)
    decide answer
    respond
}
```

---

## Memory Types

AION provides four built-in memory types:

| Type | Description | Use Case |
|------|-------------|----------|
| `working` | Short-term, limited capacity | Current task context |
| `episodic` | Event-based, temporal | Session history |
| `long_term` | Persistent storage | Important information |
| `semantic` | Factual knowledge | Concepts and facts |

### Memory Operations

```aion
# Store in memory
store value in memory_name

# Recall from memory
recall from memory_name where condition
```

---

## Models

Models represent LLM connections.

### Providers

- `openai` - OpenAI API (GPT-3.5, GPT-4)
- `anthropic` - Anthropic API (Claude)
- `ollama` - Local Ollama server
- `local` - Alias for Ollama

### Configuration

```aion
model LLM {
  provider = "openai"
  name = "gpt-4"
  api_key = "sk-..."    # Optional, uses env var
}
```

---

## Tools

Tools are external capabilities with trust and cost governance.

### Trust Levels

| Level | Description |
|-------|-------------|
| `low` | Sandboxed, limited access |
| `medium` | Standard access with logging |
| `high` | Full access, minimal restrictions |

### Cost Levels

| Level | Description |
|-------|-------------|
| `low` | Free or negligible cost |
| `medium` | Moderate cost per use |
| `high` | Expensive, use sparingly |

### Using Tools

```aion
use tool_name(arg1, arg2)
```

---

## Policies

Policies enforce safety and resource constraints.

### Common Policies

```aion
policy {
  max_tokens = 2048       # Maximum tokens per response
  allow_web = true        # Allow web access
  require_approval = false # Require human approval
}
```

---

## Reasoning Blocks

First-class cognitive operations:

### `think`

Initiates internal reasoning.

```aion
think                           # General thinking
think "Consider the options"    # Guided thinking
```

### `analyze`

Examines a subject in detail.

```aion
analyze question
analyze user_input
```

### `reflect`

Introspective evaluation.

```aion
reflect                    # General reflection
reflect on past_actions    # Specific reflection
```

### `decide`

Makes a choice with reasoning.

```aion
decide answer
decide best_option
```

---

## Control Flow

### If/Else

```aion
if condition:
  # then block
else:
  # else block
```

### When

Conditional execution (reactive).

```aion
when needs_search:
  use web_search(query)
```

### Repeat

Loop construct.

```aion
repeat 3 times:
  # loop body

repeat:
  # infinite loop (until break condition)
```

---

## Actions

### respond

Output to the user.

```aion
respond answer
respond "Hello, world!"
```

### use

Invoke a tool.

```aion
use calculator(expression)
use web_search("AION language")
```

### emit

Trigger an event.

```aion
emit notify_user(message)
```

### store / recall

Memory operations.

```aion
store important_fact in long_term
recall from semantic where topic == "AI"
```

---

## Expressions

### Operators

| Category | Operators |
|----------|-----------|
| Arithmetic | `+`, `-`, `*`, `/` |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| Logical | `and`, `or`, `not` |

### Literals

```aion
"string"      # String
42            # Integer
3.14          # Float
true / false  # Boolean
null          # Null
[1, 2, 3]     # List
```

### Member Access

```aion
object.property
result.data.value
```

---

## Event Handlers

```aion
on input(param):
  # Handle input

on error(err):
  # Handle errors

on timeout():
  # Handle timeout

on complete():
  # Handle completion
```

---

## Comments

```aion
# This is a comment
```

---

## File Extension

AION source files use `.aion` or `.ai` extension.
