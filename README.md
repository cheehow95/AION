# AION

**Artificial Intelligence Oriented Notation**

A declarative, AI-native programming language for building thinking systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

AION is designed to make AI behavior **explicit, readable, and debuggable**. Instead of scattering AI logic across prompts, Python code, and JSON configs, AION provides a unified language where:

- **Agents** are first-class entities, not functions
- **Reasoning** is visible, not hidden in chain-of-thought
- **Memory** is structured, not just text
- **Tools** are governed, not hacked in

```aion
agent Assistant {
  goal "Answer user questions clearly"

  memory working
  memory long_term

  model LLM

  tool web_search
  tool calculator

  on input(question):
    think
    analyze question
    if needs_search:
      use web_search
    decide answer
    respond
}
```

## Installation

```bash
pip install aion

# With LLM providers
pip install aion[openai]      # OpenAI support
pip install aion[anthropic]   # Anthropic support  
pip install aion[all]         # All providers
```

## Quick Start

1. Create `hello.aion`:

```aion
model LLM {
  provider = "openai"
  name = "gpt-4"
}

agent Greeter {
  goal "Greet users"
  memory working
  model LLM
  
  on input(name):
    think
    decide greeting
    respond greeting
}
```

2. Run it:

```bash
aion hello.aion --input "World"
```

## Features

### 🧠 First-Class Reasoning

```aion
think "Analyze the problem"
analyze user_input
reflect on past_actions
decide best_response
```

### 💾 Structured Memory

```aion
memory working      # Short-term context
memory episodic     # Session history
memory long_term    # Persistent storage
memory semantic     # Facts and concepts
```

### 🔧 Governed Tools

```aion
tool web_search {
  trust = medium
  cost = low
}
```

### 🎯 Declarative Agents

```aion
agent Researcher {
  goal "Find accurate information"
  # ...capabilities and behaviors
}
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Language Reference](docs/language-reference.md)
- [Examples](examples/)

## VS Code Extension

Install syntax highlighting and snippets:

```bash
code --install-extension aion-language-0.1.0.vsix
```

## CLI Usage

```bash
aion file.aion              # Run program
aion file.aion --transpile  # Generate Python
aion file.aion --parse      # Show AST
aion file.aion --tokens     # Show tokens
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE).
