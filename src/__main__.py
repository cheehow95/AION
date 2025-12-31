"""
AION Command Line Interface
"""

import sys
import argparse
import asyncio


def main():
    parser = argparse.ArgumentParser(
        description='AION - Artificial Intelligence Oriented Notation',
        prog='aion'
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='AION source file to run'
    )
    
    parser.add_argument(
        '--transpile', '-t',
        action='store_true',
        help='Transpile to Python instead of running'
    )
    
    parser.add_argument(
        '--parse', '-p',
        action='store_true',
        help='Parse and print AST'
    )
    
    parser.add_argument(
        '--tokens', '-k',
        action='store_true',
        help='Tokenize and print tokens'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='store_true',
        help='Show version'
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input to pass to the agent'
    )
    
    args = parser.parse_args()
    
    if args.version:
        from . import __version__
        print(f"AION version {__version__}")
        return
    
    if not args.file:
        parser.print_help()
        return
    
    # Read source file
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            source = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Tokenize mode
    if args.tokens:
        from .lexer import tokenize
        try:
            tokens = tokenize(source)
            for token in tokens:
                print(token)
        except Exception as e:
            print(f"Lexer error: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Parse mode
    if args.parse:
        from .parser import parse
        try:
            ast = parse(source)
            print_ast(ast)
        except Exception as e:
            print(f"Parser error: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Transpile mode
    if args.transpile:
        from .transpiler import transpile
        try:
            python_code = transpile(source)
            print(python_code)
        except Exception as e:
            print(f"Transpiler error: {e}", file=sys.stderr)
            sys.exit(1)
        return
    
    # Run mode (default)
    from .interpreter import run_aion
    try:
        result = asyncio.run(run_aion(source, args.input))
        
        if result.get('output'):
            for line in result['output']:
                print(line)
        
        if result.get('trace'):
            print("\n--- Reasoning Trace ---")
            for step in result['trace'].get('steps', []):
                print(f"[{step['type'].upper()}] {step['input']}")
                if step.get('output'):
                    print(f"  -> {str(step['output'])[:100]}...")
    
    except Exception as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        sys.exit(1)


def print_ast(node, indent=0):
    """Pretty print an AST node."""
    prefix = "  " * indent
    node_name = node.__class__.__name__
    
    # Get relevant attributes
    attrs = []
    if hasattr(node, 'name'):
        attrs.append(f"name={node.name!r}")
    if hasattr(node, 'value'):
        attrs.append(f"value={node.value!r}")
    if hasattr(node, 'goal'):
        attrs.append(f"goal={node.goal!r}")
    if hasattr(node, 'memory_type'):
        attrs.append(f"type={node.memory_type!r}")
    if hasattr(node, 'operator'):
        attrs.append(f"op={node.operator!r}")
    
    attrs_str = ", ".join(attrs) if attrs else ""
    print(f"{prefix}{node_name}({attrs_str})")
    
    # Print children
    for attr in ['declarations', 'body', 'then_body', 'else_body', 
                 'left', 'right', 'operand', 'target', 'elements',
                 'condition', 'value', 'args']:
        if hasattr(node, attr):
            child = getattr(node, attr)
            if isinstance(child, list):
                for item in child:
                    if hasattr(item, '__class__') and hasattr(item, 'accept'):
                        print_ast(item, indent + 1)
            elif hasattr(child, '__class__') and hasattr(child, 'accept'):
                print_ast(child, indent + 1)


if __name__ == '__main__':
    main()
