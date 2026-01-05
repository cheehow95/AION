# AION Installation Guide

Complete guide to install AION on your PC.

## 📋 Requirements

- **Python**: 3.10 or higher
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 2GB free space

## 🚀 Quick Installation

### Option 1: Clone from GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/cheehow95/AION.git
cd AION

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Option 2: Download ZIP

1. Go to https://github.com/cheehow95/AION
2. Click "Code" → "Download ZIP"
3. Extract to your desired location
4. Open terminal in the extracted folder
5. Follow steps above from "Create virtual environment"

## 📦 Dependencies

### Core Dependencies (automatically installed)
```
aiohttp          # Async HTTP client
numpy            # Numerical computing
```

### Optional Dependencies

```bash
# For LLM providers
pip install openai         # OpenAI GPT models
pip install anthropic      # Claude models
pip install ollama         # Local Ollama models

# For advanced features
pip install chromadb       # Vector database
pip install sentence-transformers  # Embeddings
```

## ✅ Verify Installation

```bash
# Run tests
python run_tests.py

# Run demo
python demo.py

# Start REPL
python repl.py
```

Expected output:
```
✅ AION v4.0 - All Systems Operational
✓ Lexer: Ready
✓ Parser: Ready
✓ Interpreter: Ready
✓ 26 Domain Engines: Ready
```

## 🎮 Quick Start

### Run an AION program

```bash
python -m aion examples/my_first_agent.aion
```

### Interactive REPL

```bash
python repl.py
```

```
AION REPL v2.0
>>> agent Greeter { goal "Say hello" }
>>> run Greeter
Hello! I'm ready to help.
```

### Transpile to Python

```bash
python -m aion examples/assistant.aion --transpile
```

## 🧪 Run All Tests

```bash
python run_tests.py
```

Expected: `All tests passed! ✓`

## 🌐 Optional: Internet Learning

To enable internet knowledge learning:

```bash
# Install async HTTP (already included)
pip install aiohttp

# Run continuous learner
python -c "from src.learning import ContinuousLearner; print('Ready!')"
```

## 🔧 VS Code Extension

1. Navigate to the extension folder:
   ```bash
   cd vscode-aion
   ```

2. Package the extension:
   ```bash
   npm install
   npx vsce package
   ```

3. Install in VS Code:
   ```bash
   code --install-extension aion-language-0.1.0.vsix
   ```

Features:
- Syntax highlighting for `.aion` files
- Code snippets
- Bracket matching

## 💻 IDE Setup

### VS Code
1. Install the AION extension (see above)
2. Open any `.aion` file

### PyCharm
1. Associate `.aion` files with Python syntax
2. Settings → Editor → File Types → Python → Add `*.aion`

## 🐛 Troubleshooting

### "Module not found" error
```bash
# Ensure virtual environment is activated
# Windows:
venv\Scripts\activate
# Then reinstall:
pip install -e .
```

### Python version error
```bash
# Check Python version
python --version
# Must be 3.10 or higher
```

### Permission errors (Linux/macOS)
```bash
chmod +x run_tests.py
python run_tests.py
```

## 📚 What's Included

```
AION/
├── src/                 # Source code
│   ├── lexer/          # Tokenizer
│   ├── parser/         # AST generator
│   ├── interpreter/    # Execution engine
│   ├── domains/        # 26 scientific engines
│   ├── learning/       # Internet learning
│   ├── consciousness/  # Meta-cognition
│   └── ...
├── examples/           # 16 AION examples
├── tests/              # Test suite
├── grammar/            # EBNF grammar
├── vscode-aion/        # VS Code extension
└── docs/               # Documentation
```

## 🎯 Next Steps

1. **Explore Examples**: Check `examples/` folder
2. **Read Docs**: See `docs/` for language reference
3. **Run Demo**: `python demo.py`
4. **Build Your First Agent**: Create a `.aion` file!

## 📞 Support

- GitHub Issues: https://github.com/cheehow95/AION/issues
- Documentation: `docs/` folder

---

**AION - Artificial Intelligence Oriented Notation**

*Think in agents. Code in AION.*
