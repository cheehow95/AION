# Getting Started with AION

## What is AION?

AION (Artificial Intelligence Oriented Notation) is a declarative programming language designed specifically for building AI systems. Instead of writing algorithms with loops and conditionals, you describe **thinking systems** with agents, memory, and reasoning.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/aion-lang/aion.git
cd aion

# Install the package
pip install -e .

# Optional: Install LLM provider packages
pip install openai anthropic httpx
```

### 2. Your First Agent

Create a file called `hello.aion`:

```aion
model LLM {
  provider = "openai"
  name = "gpt-4"
}

agent Greeter {
  goal "Greet users warmly"
  
  memory working
  model LLM
  
  on input(name):
    think "How should I greet this person?"
    decide greeting
    respond greeting
}
```

### 3. Run It

```bash
python -m aion hello.aion --input "Alice"
```

## Core Concepts

### Agents

Agents are autonomous reasoning units. They have:
- A **goal** (primary objective)
- **Memory** (different types for different purposes)
- **Models** (LLM connections)
- **Tools** (external capabilities)

```aion
agent Assistant {
  goal "Help users with their questions"
  
  memory working
  memory long_term
  
  model LLM
  tool calculator
  
  on input(question):
    think
    analyze question
    decide answer
    respond
}
```

### Reasoning Blocks

Unlike traditional code, AION makes reasoning explicit:

| Construct | Purpose |
|-----------|---------|
| `think` | Internal reasoning |
| `analyze` | Examine something in detail |
| `reflect` | Introspection |
| `decide` | Make a choice |

### Memory Types

| Type | Use For |
|------|---------|
| `working` | Current task context |
| `episodic` | Session history |
| `long_term` | Important information |
| `semantic` | Facts and concepts |

## CLI Commands

```bash
# Run an AION program
python -m aion program.aion

# Pass input to the agent
python -m aion program.aion --input "Hello"

# Transpile to Python
python -m aion program.aion --transpile

# Parse and show AST
python -m aion program.aion --parse

# Tokenize source
python -m aion program.aion --tokens
```

## VS Code Extension

Install the AION extension for syntax highlighting:

1. Open VS Code Extensions
2. Search for "AION Language"
3. Click Install

Or install from the command line:
```bash
code --install-extension aion-language-0.1.0.vsix
```

## Example Programs

### Simple Calculator Agent

```aion
tool calculator {
  trust = high
  cost = low
}

agent MathHelper {
  goal "Help with math problems"
  
  memory working
  tool calculator
  
  on input(problem):
    analyze problem
    use calculator(expression)
    respond result
}
```

### Research Agent

```aion
tool web_search {
  trust = medium
  cost = medium
}

agent Researcher {
  goal "Find accurate information"
  
  memory working
  memory long_term
  
  tool web_search
  
  on input(query):
    think "What do I need to find?"
    use web_search(query)
    analyze results
    store findings in long_term
    decide answer
    respond
}
```

## Next Steps

1. Read the [Language Reference](./language-reference.md)
2. Explore the [examples/](../examples/) directory
3. Build your own agents!

## Getting Help

- GitHub Issues: [github.com/aion-lang/aion/issues](https://github.com/aion-lang/aion/issues)
- Documentation: [aion-lang.dev/docs](https://aion-lang.dev/docs)
