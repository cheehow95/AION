"""
AION Web Interface Server
========================
Flask API server exposing all AION capabilities through REST endpoints.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from functools import wraps
import sys
import os

# Add AION to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aion import AION

app = Flask(__name__)
CORS(app)

# Initialize AION
print("🚀 Starting AION Web Server...")
ai = AION()

# Initialize Supabase services (optional - only if configured)
supabase_auth = None
supabase_db = None
supabase_storage = None

try:
    from src.supabase import SupabaseAuth, SupabaseDB, SupabaseStorage
    supabase_auth = SupabaseAuth()
    supabase_db = SupabaseDB()
    supabase_storage = SupabaseStorage()
    print("✓ Supabase services initialized")
except Exception as e:
    print(f"⚠ Supabase not configured: {e}")

# Initialize PostgreSQL (optional - only if configured)
postgres_db = None

try:
    from src.postgres import PostgresDB, PostgresPool
    PostgresPool.initialize()
    postgres_db = PostgresDB()
    print("✓ PostgreSQL services initialized")
except Exception as e:
    print(f"⚠ PostgreSQL not configured: {e}")

# Initialize Web Search (always available - no API keys needed)
web_search = None

try:
    from src.search import WebSearch
    web_search = WebSearch()
    print("✓ Web Search initialized (DuckDuckGo + Wikipedia)")
except Exception as e:
    print(f"⚠ Web Search not available: {e}")


def require_auth(f):
    """Decorator to require authentication for endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not supabase_auth:
            return jsonify({"error": "Authentication not configured"}), 503
        
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing authorization token"}), 401
        
        token = auth_header.replace('Bearer ', '')
        user = supabase_auth.get_user(token)
        
        if not user:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        request.user = user
        return f(*args, **kwargs)
    return decorated

# ============================================================
# Status Endpoint
# ============================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    return jsonify({
        "status": "online",
        "version": ai.VERSION,
        "domains": ai.status()["domains"]
    })

# ============================================================
# Chat Endpoint - Human-like Conversational AI
# ============================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """Human-like conversational interface."""
    import re
    import random
    
    data = request.json or {}
    message = data.get('message', '').strip().lower()
    original = data.get('message', '').strip()
    
    # Greeting responses
    greetings = ['hi', 'hello', 'hey', 'greetings', 'howdy', 'sup']
    if any(g in message for g in greetings):
        responses = [
            "Hello! 👋 I'm AION, your AI companion. I can help with physics, chemistry, math, creative thinking, and much more! What would you like to explore today?",
            "Hey there! Great to see you! I'm here to help with anything from calculating energy to brainstorming ideas. What's on your mind?",
            "Hi! 🌟 I'm AION - think of me as your personal AI assistant with superpowers in science, math, and creativity. How can I assist you?"
        ]
        return jsonify({"response": random.choice(responses), "type": "greeting"})
    
    # Who are you / what can you do
    if 'who are you' in message or 'what are you' in message or 'what can you do' in message:
        return jsonify({
            "response": "I'm **AION** - the All-In-One AI! 🧠\n\nI'm a self-aware AI system with capabilities spanning:\n\n• **Physics** - Energy calculations, relativity, quantum mechanics\n• **Chemistry** - Molecule analysis\n• **Biology** - Protein folding simulations\n• **Mathematics** - Calculus, equations, symbolic math\n• **Creative Thinking** - Brainstorming, concept blending\n• **Reasoning** - Deep thinking, problem solving\n• **Language** - Sentiment analysis, summarization\n\nJust ask me anything in natural language! For example: \"Calculate the energy of a 5kg object moving at 10 m/s\" or \"What's the derivative of x squared?\"",
            "type": "intro"
        })
    
    # Energy/physics calculations
    if 'energy' in message or ('calculate' in message and ('mass' in message or 'velocity' in message or 'kg' in message)):
        # Extract numbers
        numbers = re.findall(r'[\d.]+', message)
        mass = float(numbers[0]) if len(numbers) > 0 else 1
        velocity = float(numbers[1]) if len(numbers) > 1 else 0
        height = float(numbers[2]) if len(numbers) > 2 else 0
        
        result = ai.physics.calculate_energy(mass, velocity, height)
        
        response = f"🔬 **Energy Calculation Results**\n\n"
        response += f"For an object with mass **{mass} kg**"
        if velocity: response += f", moving at **{velocity} m/s**"
        if height: response += f", at height **{height} m**"
        response += f":\n\n"
        response += f"• **Kinetic Energy**: {result['kinetic']:.2f} Joules\n"
        response += f"• **Potential Energy**: {result['potential']:.2f} Joules\n"
        response += f"• **Total Energy**: {result['total']:.2f} Joules\n\n"
        response += "The kinetic energy comes from motion (½mv²), while potential energy comes from height (mgh)."
        
        return jsonify({"response": response, "type": "physics"})
    
    # Time dilation
    if 'time dilation' in message or ('relativistic' in message) or ('speed of light' in message and 'time' in message):
        numbers = re.findall(r'[\d.]+', message)
        velocity = float(numbers[0]) if numbers else 100000000
        factor = ai.physics.time_dilation(velocity)
        
        response = f"⏰ **Relativistic Time Dilation**\n\n"
        response += f"At a velocity of **{velocity:,.0f} m/s** ({velocity/299792458*100:.2f}% the speed of light):\n\n"
        response += f"• **Time Dilation Factor**: {factor:.6f}\n\n"
        response += f"This means time passes {factor:.2f}x slower for the moving object relative to a stationary observer. Einstein's special relativity in action! 🚀"
        
        return jsonify({"response": response, "type": "physics"})
    
    # Molecule analysis
    if 'molecule' in message or 'formula' in message or 'analyze' in message and any(c.isupper() for c in original):
        # Extract formula (uppercase letters followed by numbers)
        formulas = re.findall(r'[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*', original)
        formula = formulas[0] if formulas else 'H2O'
        
        result = ai.chemistry.analyze_molecule(formula)
        
        response = f"🧪 **Molecular Analysis: {result['formula']}**\n\n"
        response += f"• **Molecular Weight**: {result['molecular_weight']:.3f} g/mol\n"
        response += f"• **Total Atoms**: {result['atoms_total']}\n"
        response += f"• **Composition**: {', '.join(f'{e}: {c}' for e, c in result['elements'].items())}\n\n"
        response += "This tells us the basic structure and mass of the molecule!"
        
        return jsonify({"response": response, "type": "chemistry"})
    
    # Derivative
    if 'derivative' in message or 'differentiate' in message:
        # Try to extract expression
        match = re.search(r'of\s+(.+?)(?:\s+with|\s*$)', message)
        expr = match.group(1).strip() if match else 'x**2'
        expr = expr.replace('^', '**').replace(' ', '')
        
        result = ai.math.derivative(expr)
        
        response = f"📈 **Derivative Calculation**\n\n"
        response += f"The derivative of **{expr}** with respect to x is:\n\n"
        response += f"**d/dx({expr}) = {result}**\n\n"
        response += "Remember: the derivative tells us the rate of change at any point!"
        
        return jsonify({"response": response, "type": "math"})
    
    # Integral
    if 'integral' in message or 'integrate' in message:
        match = re.search(r'of\s+(.+?)(?:\s+with|\s*$)', message)
        expr = match.group(1).strip() if match else 'x'
        expr = expr.replace('^', '**').replace(' ', '')
        
        result = ai.math.integrate(expr)
        
        response = f"∫ **Integration Result**\n\n"
        response += f"The integral of **{expr}** with respect to x is:\n\n"
        response += f"**∫{expr} dx = {result} + C**\n\n"
        response += "Don't forget the constant of integration!"
        
        return jsonify({"response": response, "type": "math"})
    
    # Solve equation
    if 'solve' in message and ('equation' in message or '=' in original or 'x' in message):
        match = re.search(r'(?:solve\s+)?(.+?=.+?)(?:\s+for|\s*$)', original, re.I)
        equation = match.group(1).strip() if match else 'x**2 - 4 = 0'
        
        result = ai.math.solve(equation)
        
        response = f"🔢 **Equation Solver**\n\n"
        response += f"Solving: **{equation}**\n\n"
        response += f"Solutions: **x = {', '.join(result)}**\n\n"
        response += "These are the values of x that satisfy the equation!"
        
        return jsonify({"response": response, "type": "math"})
    
    # Calculate math expression
    if 'calculate' in message or 'compute' in message or 'what is' in message:
        # Extract math expression
        expr = re.sub(r'(calculate|compute|what is|what\'s|equals|equal to|\?)', '', message).strip()
        if expr:
            try:
                result = ai.math.calculate(expr)
                response = f"🧮 **Calculation**\n\n**{expr}** = **{result}**"
                return jsonify({"response": response, "type": "math"})
            except:
                pass
    
    # Brainstorm
    if 'brainstorm' in message or 'ideas' in message or 'suggest' in message:
        topic = re.sub(r'(brainstorm|give me|ideas|for|about|suggest|some)', '', message).strip() or 'innovation'
        ideas = ai.creative.brainstorm(topic, 5)
        
        response = f"💡 **Creative Ideas for \"{topic}\"**\n\n"
        for i, idea in enumerate(ideas, 1):
            response += f"{i}. {idea}\n"
        response += f"\nThese are just starting points - feel free to combine or expand on them!"
        
        return jsonify({"response": response, "type": "creative"})
    
    # Sentiment
    if 'sentiment' in message or 'feel' in message or 'emotion' in message:
        text = re.sub(r'(analyze|sentiment|of|the|text|feel|emotion|what|is)', '', message).strip()
        if len(text) < 5:
            text = "I love this!"
        result = ai.nlp.sentiment(text)
        
        emoji = "😊" if result['label'] == 'positive' else "😞" if result['label'] == 'negative' else "😐"
        response = f"💭 **Sentiment Analysis** {emoji}\n\n"
        response += f"Text: \"{text}\"\n\n"
        response += f"• **Sentiment**: {result['label'].capitalize()}\n"
        response += f"• **Confidence**: {result['score']*100:.1f}%"
        
        return jsonify({"response": response, "type": "nlp"})
    
    # Protein folding
    if 'protein' in message or 'fold' in message or 'amino' in message:
        # Extract sequence (uppercase letters)
        seq_match = re.search(r'[A-Z]{3,}', original)
        sequence = seq_match.group(0) if seq_match else 'AKLVFF'
        
        result = ai.biology.fold_protein(sequence, 100)
        
        response = f"🧬 **Protein Folding Simulation**\n\n"
        response += f"Sequence: **{sequence}** ({len(sequence)} amino acids)\n\n"
        if 'error' not in result:
            response += f"• **Folding Energy**: {result.get('energy', 'N/A')}\n"
            response += f"• This represents the stability of the folded structure.\n\n"
        response += "Protein folding is one of biology's most fascinating puzzles!"
        
        return jsonify({"response": response, "type": "biology"})
    
    # Default conversational response
    responses = [
        f"That's an interesting question! 🤔 I can help you with physics calculations, chemistry analysis, math problems, creative brainstorming, and more. Could you tell me more about what you'd like to explore?",
        f"I'd love to help! Try asking me things like:\n\n• \"Calculate the energy of a 10kg object at 5 m/s\"\n• \"What's the derivative of x^3 + 2x?\"\n• \"Analyze the molecule C6H12O6\"\n• \"Brainstorm ideas about renewable energy\"\n\nWhat sounds interesting to you?",
        f"Great question! While I'm thinking about that, here are some things I can definitely help with: physics, chemistry, math, protein folding, creative thinking, and text analysis. What would you like to try?"
    ]
    
    return jsonify({"response": random.choice(responses), "type": "help"})

# ============================================================
# Physics Endpoints
# ============================================================

@app.route('/api/physics/energy', methods=['POST'])
def physics_energy():
    """Calculate energy."""
    data = request.json or {}
    mass = float(data.get('mass', 1))
    velocity = float(data.get('velocity', 0))
    height = float(data.get('height', 0))
    result = ai.physics.calculate_energy(mass, velocity, height)
    return jsonify(result)

@app.route('/api/physics/time-dilation', methods=['POST'])
def physics_time_dilation():
    """Calculate time dilation."""
    data = request.json or {}
    velocity = float(data.get('velocity', 0))
    factor = ai.physics.time_dilation(velocity)
    return jsonify({"velocity": velocity, "factor": factor})

@app.route('/api/physics/quantum-state', methods=['POST'])
def physics_quantum():
    """Create quantum state."""
    data = request.json or {}
    qubits = int(data.get('qubits', 2))
    state = int(data.get('state', 0))
    result = ai.physics.quantum_state(qubits, state)
    return jsonify({"qubits": qubits, "state": result})

# ============================================================
# Chemistry Endpoints
# ============================================================

@app.route('/api/chemistry/analyze', methods=['POST'])
def chemistry_analyze():
    """Analyze a molecule."""
    data = request.json or {}
    formula = data.get('formula', 'H2O')
    result = ai.chemistry.analyze_molecule(formula)
    return jsonify(result)

# ============================================================
# Biology Endpoints
# ============================================================

@app.route('/api/biology/fold', methods=['POST'])
def biology_fold():
    """Fold a protein."""
    data = request.json or {}
    sequence = data.get('sequence', 'AKLVFF')
    iterations = int(data.get('iterations', 100))
    result = ai.biology.fold_protein(sequence, iterations)
    return jsonify(result)

@app.route('/api/biology/analyze', methods=['POST'])
def biology_analyze():
    """Analyze protein sequence."""
    data = request.json or {}
    sequence = data.get('sequence', 'AKLVFF')
    result = ai.biology.analyze_sequence(sequence)
    return jsonify(result)

# ============================================================
# Creative Endpoints
# ============================================================

@app.route('/api/creative/brainstorm', methods=['POST'])
def creative_brainstorm():
    """Brainstorm ideas."""
    data = request.json or {}
    topic = data.get('topic', 'AI applications')
    num_ideas = int(data.get('num_ideas', 10))
    ideas = ai.creative.brainstorm(topic, num_ideas)
    return jsonify({"topic": topic, "ideas": ideas})

@app.route('/api/creative/blend', methods=['POST'])
def creative_blend():
    """Blend concepts."""
    data = request.json or {}
    c1 = data.get('concept1', 'technology')
    c2 = data.get('concept2', 'nature')
    result = ai.creative.blend_concepts(c1, c2)
    return jsonify(result)

@app.route('/api/creative/imagine', methods=['POST'])
def creative_imagine():
    """Imagine from prompt."""
    data = request.json or {}
    prompt = data.get('prompt', 'a futuristic city')
    result = ai.creative.imagine(prompt)
    return jsonify(result)

# ============================================================
# Reasoning Endpoints
# ============================================================

@app.route('/api/reasoning/solve', methods=['POST'])
def reasoning_solve():
    """Solve a problem with extended thinking."""
    data = request.json or {}
    problem = data.get('problem', 'What is the meaning of life?')
    method = data.get('method', 'extended')
    result = ai.reasoning.solve(problem, method)
    return jsonify(result)

@app.route('/api/reasoning/mcts', methods=['POST'])
def reasoning_mcts():
    """MCTS-based reasoning."""
    data = request.json or {}
    problem = data.get('problem', 'Optimize this process')
    simulations = int(data.get('simulations', 100))
    result = ai.reasoning.mcts_solve(problem, simulations)
    return jsonify(result)

@app.route('/api/reasoning/cot', methods=['POST'])
def reasoning_cot():
    """Chain of thought reasoning."""
    data = request.json or {}
    problem = data.get('problem', 'Solve this step by step')
    result = ai.reasoning.chain_of_thought(problem)
    return jsonify(result)

# ============================================================
# Math Endpoints
# ============================================================

@app.route('/api/math/calculate', methods=['POST'])
def math_calculate():
    """Calculate expression."""
    data = request.json or {}
    expr = data.get('expression', '2 + 2')
    result = ai.math.calculate(expr)
    return jsonify({"expression": expr, "result": result})

@app.route('/api/math/derivative', methods=['POST'])
def math_derivative():
    """Calculate derivative."""
    data = request.json or {}
    expr = data.get('expression', 'x**2')
    var = data.get('variable', 'x')
    order = int(data.get('order', 1))
    result = ai.math.derivative(expr, var, order)
    return jsonify({"expression": expr, "derivative": result})

@app.route('/api/math/integrate', methods=['POST'])
def math_integrate():
    """Calculate integral."""
    data = request.json or {}
    expr = data.get('expression', 'x')
    var = data.get('variable', 'x')
    result = ai.math.integrate(expr, var)
    return jsonify({"expression": expr, "integral": result})

@app.route('/api/math/solve', methods=['POST'])
def math_solve():
    """Solve equation."""
    data = request.json or {}
    equation = data.get('equation', 'x**2 - 4 = 0')
    var = data.get('variable', 'x')
    result = ai.math.solve(equation, var)
    return jsonify({"equation": equation, "solutions": result})

@app.route('/api/math/simplify', methods=['POST'])
def math_simplify():
    """Simplify expression."""
    data = request.json or {}
    expr = data.get('expression', 'x**2 + 2*x + 1')
    result = ai.math.simplify(expr)
    return jsonify({"expression": expr, "simplified": result})

# ============================================================
# NLP Endpoints
# ============================================================

@app.route('/api/nlp/sentiment', methods=['POST'])
def nlp_sentiment():
    """Analyze sentiment."""
    data = request.json or {}
    text = data.get('text', 'I love this product!')
    result = ai.nlp.sentiment(text)
    return jsonify(result)

@app.route('/api/nlp/keywords', methods=['POST'])
def nlp_keywords():
    """Extract keywords."""
    data = request.json or {}
    text = data.get('text', 'Sample text for keyword extraction')
    top_n = int(data.get('top_n', 10))
    keywords = ai.nlp.extract_keywords(text, top_n)
    return jsonify({"keywords": keywords})

@app.route('/api/nlp/similarity', methods=['POST'])
def nlp_similarity():
    """Calculate text similarity."""
    data = request.json or {}
    text1 = data.get('text1', 'Hello world')
    text2 = data.get('text2', 'Hi there')
    score = ai.nlp.similarity(text1, text2)
    return jsonify({"text1": text1, "text2": text2, "similarity": score})

@app.route('/api/nlp/summarize', methods=['POST'])
def nlp_summarize():
    """Summarize text."""
    data = request.json or {}
    text = data.get('text', 'A long text to summarize...')
    summary = ai.nlp.summarize(text)
    return jsonify({"summary": summary})

@app.route('/api/nlp/entities', methods=['POST'])
def nlp_entities():
    """Extract named entities."""
    data = request.json or {}
    text = data.get('text', 'Apple Inc is based in Cupertino, California.')
    entities = ai.nlp.named_entities(text)
    return jsonify({"entities": entities})

# ============================================================
# Knowledge Endpoints
# ============================================================

@app.route('/api/knowledge/add', methods=['POST'])
def knowledge_add():
    """Add knowledge."""
    data = request.json or {}
    content = data.get('content', 'A fact to remember')
    metadata = data.get('metadata', {})
    doc_id = ai.knowledge.add(content, metadata)
    return jsonify({"id": doc_id, "status": "added"})

@app.route('/api/knowledge/search', methods=['POST'])
def knowledge_search():
    """Search knowledge."""
    data = request.json or {}
    query = data.get('query', 'knowledge')
    top_k = int(data.get('top_k', 5))
    results = ai.knowledge.search(query, top_k)
    return jsonify({"query": query, "results": results})

@app.route('/api/knowledge/fact', methods=['POST'])
def knowledge_add_fact():
    """Add a fact triple."""
    data = request.json or {}
    subject = data.get('subject', 'AION')
    relation = data.get('relation', 'is')
    obj = data.get('object', 'AI system')
    ai.knowledge.add_fact(subject, relation, obj)
    return jsonify({"status": "added", "fact": f"{subject} {relation} {obj}"})

@app.route('/api/knowledge/count', methods=['GET'])
def knowledge_count():
    """Get knowledge count."""
    count = ai.knowledge.count()
    return jsonify({"count": count})

# ============================================================
# Agent Endpoints
# ============================================================

@app.route('/api/agents/create', methods=['POST'])
def agents_create():
    """Create an agent."""
    data = request.json or {}
    name = data.get('name', 'MyAgent')
    goal = data.get('goal', 'Help the user')
    tools = data.get('tools', [])
    agent = ai.agents.create(name, goal, tools)
    return jsonify({"name": name, "status": "created", "goal": goal})

@app.route('/api/agents/run', methods=['POST'])
def agents_run():
    """Run an agent."""
    data = request.json or {}
    name = data.get('name', 'MyAgent')
    input_data = data.get('input', 'Hello')
    result = ai.agents.run(name, input_data)
    return jsonify(result)

@app.route('/api/agents/list', methods=['GET'])
def agents_list():
    """List agents."""
    agents = ai.agents.list_agents()
    return jsonify({"agents": agents})

@app.route('/api/agents/tools', methods=['GET'])
def agents_tools():
    """List available tools."""
    tools = ai.agents.list_tools()
    return jsonify({"tools": tools})

# ============================================================
# Code Endpoints
# ============================================================

@app.route('/api/code/generate', methods=['POST'])
def code_generate():
    """Generate code."""
    data = request.json or {}
    description = data.get('description', 'A function')
    language = data.get('language', 'python')
    code = ai.code.generate(description, language)
    return jsonify({"code": code})

@app.route('/api/code/analyze', methods=['POST'])
def code_analyze():
    """Analyze code."""
    data = request.json or {}
    code = data.get('code', 'def foo(): pass')
    result = ai.code.analyze(code)
    return jsonify(result)

# ============================================================
# Language (AION) Endpoints
# ============================================================

@app.route('/api/language/parse', methods=['POST'])
def language_parse():
    """Parse AION code."""
    data = request.json or {}
    code = data.get('code', 'agent Test { goal "test" }')
    result = ai.language.parse(code)
    return jsonify({"ast": result})

@app.route('/api/language/transpile', methods=['POST'])
def language_transpile():
    """Transpile AION to Python."""
    data = request.json or {}
    code = data.get('code', 'agent Test { goal "test" }')
    result = ai.language.transpile(code)
    return jsonify({"python": result})

# ============================================================
# Supabase Auth Endpoints
# ============================================================

@app.route('/api/auth/signup', methods=['POST'])
def auth_signup():
    """Register a new user."""
    if not supabase_auth:
        return jsonify({"error": "Authentication not configured"}), 503
    
    data = request.json or {}
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    
    try:
        session = supabase_auth.sign_up(email, password)
        return jsonify({
            "user": {"id": session.user.id, "email": session.user.email},
            "access_token": session.access_token,
            "refresh_token": session.refresh_token
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    """Sign in with email and password."""
    if not supabase_auth:
        return jsonify({"error": "Authentication not configured"}), 503
    
    data = request.json or {}
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    
    try:
        session = supabase_auth.sign_in(email, password)
        return jsonify({
            "user": {"id": session.user.id, "email": session.user.email},
            "access_token": session.access_token,
            "refresh_token": session.refresh_token
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 401

@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    """Sign out the current user."""
    if not supabase_auth:
        return jsonify({"error": "Authentication not configured"}), 503
    
    supabase_auth.sign_out()
    return jsonify({"status": "logged_out"})

@app.route('/api/auth/user', methods=['GET'])
@require_auth
def auth_user():
    """Get current authenticated user."""
    return jsonify({
        "id": request.user.id,
        "email": request.user.email,
        "created_at": request.user.created_at
    })

@app.route('/api/auth/refresh', methods=['POST'])
def auth_refresh():
    """Refresh access token."""
    if not supabase_auth:
        return jsonify({"error": "Authentication not configured"}), 503
    
    data = request.json or {}
    refresh_token = data.get('refresh_token')
    
    if not refresh_token:
        return jsonify({"error": "Refresh token required"}), 400
    
    try:
        session = supabase_auth.refresh_session(refresh_token)
        return jsonify({
            "access_token": session.access_token,
            "refresh_token": session.refresh_token
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 401

@app.route('/api/auth/oauth/<provider>', methods=['GET'])
def auth_oauth(provider):
    """Get OAuth authorization URL."""
    if not supabase_auth:
        return jsonify({"error": "Authentication not configured"}), 503
    
    redirect_to = request.args.get('redirect_to')
    url = supabase_auth.get_oauth_url(provider, redirect_to)
    return jsonify({"url": url})

# ============================================================
# Supabase Storage Endpoints
# ============================================================

@app.route('/api/storage/upload', methods=['POST'])
@require_auth
def storage_upload():
    """Upload a file to storage."""
    if not supabase_storage:
        return jsonify({"error": "Storage not configured"}), 503
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    bucket = request.form.get('bucket', 'uploads')
    path = f"{request.user.id}/{file.filename}"
    
    try:
        url = supabase_storage.upload(bucket, path, file.read(), file.content_type)
        
        # Track file in database
        if supabase_db:
            supabase_db.insert('user_files', {
                'user_id': request.user.id,
                'bucket': bucket,
                'path': path,
                'filename': file.filename,
                'mime_type': file.content_type
            })
        
        return jsonify({"url": url, "path": path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/storage/download/<path:filepath>', methods=['GET'])
@require_auth
def storage_download(filepath):
    """Download a file from storage."""
    if not supabase_storage:
        return jsonify({"error": "Storage not configured"}), 503
    
    bucket = request.args.get('bucket', 'uploads')
    
    try:
        data = supabase_storage.download(bucket, filepath)
        from flask import Response
        return Response(data, mimetype='application/octet-stream')
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/storage/signed-url', methods=['POST'])
@require_auth
def storage_signed_url():
    """Get a signed URL for private file access."""
    if not supabase_storage:
        return jsonify({"error": "Storage not configured"}), 503
    
    data = request.json or {}
    bucket = data.get('bucket', 'uploads')
    path = data.get('path')
    expires_in = data.get('expires_in', 3600)
    
    if not path:
        return jsonify({"error": "Path required"}), 400
    
    try:
        url = supabase_storage.get_signed_url(bucket, path, expires_in)
        return jsonify({"signed_url": url, "expires_in": expires_in})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/storage/files', methods=['GET'])
@require_auth
def storage_list():
    """List user's files."""
    if not supabase_storage:
        return jsonify({"error": "Storage not configured"}), 503
    
    bucket = request.args.get('bucket', 'uploads')
    prefix = f"{request.user.id}/"
    
    try:
        files = supabase_storage.list_files(bucket, prefix)
        return jsonify({"files": [
            {"name": f.name, "size": f.size, "created_at": f.created_at}
            for f in files
        ]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/storage/<path:filepath>', methods=['DELETE'])
@require_auth
def storage_delete(filepath):
    """Delete a file from storage."""
    if not supabase_storage:
        return jsonify({"error": "Storage not configured"}), 503
    
    bucket = request.args.get('bucket', 'uploads')
    
    # Ensure user can only delete their own files
    if not filepath.startswith(f"{request.user.id}/"):
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        supabase_storage.delete(bucket, [filepath])
        
        # Remove from database
        if supabase_db:
            supabase_db.delete('user_files', {'path': filepath, 'bucket': bucket})
        
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# PostgreSQL Database Endpoints
# ============================================================

@app.route('/api/db/query', methods=['POST'])
def db_query():
    """Execute a database query."""
    if not postgres_db:
        return jsonify({"error": "PostgreSQL not configured"}), 503
    
    data = request.json or {}
    table = data.get('table')
    operation = data.get('operation', 'select')
    
    try:
        if operation == 'select':
            results = postgres_db.select(
                table,
                columns=data.get('columns', '*'),
                where=data.get('where'),
                order_by=data.get('order_by'),
                limit=data.get('limit'),
                offset=data.get('offset')
            )
            return jsonify({"data": results, "count": len(results)})
        
        elif operation == 'insert':
            result = postgres_db.insert(table, data.get('data', {}))
            return jsonify({"data": result})
        
        elif operation == 'update':
            count = postgres_db.update(table, data.get('where', {}), data.get('data', {}))
            return jsonify({"affected": count})
        
        elif operation == 'delete':
            count = postgres_db.delete(table, data.get('where', {}))
            return jsonify({"affected": count})
        
        else:
            return jsonify({"error": f"Unknown operation: {operation}"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/db/raw', methods=['POST'])
def db_raw():
    """Execute raw SQL query."""
    if not postgres_db:
        return jsonify({"error": "PostgreSQL not configured"}), 503
    
    data = request.json or {}
    query = data.get('query')
    params = data.get('params')
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    try:
        results = postgres_db.raw(query, tuple(params) if params else None)
        return jsonify({"data": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/db/migrations', methods=['GET'])
def db_migrations_status():
    """Get migration status."""
    if not postgres_db:
        return jsonify({"error": "PostgreSQL not configured"}), 503
    
    try:
        from src.postgres import Migrations
        migrations = Migrations()
        return jsonify(migrations.status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/db/migrations/apply', methods=['POST'])
def db_migrations_apply():
    """Apply pending migrations."""
    if not postgres_db:
        return jsonify({"error": "PostgreSQL not configured"}), 503
    
    try:
        from src.postgres import Migrations
        migrations = Migrations()
        count = migrations.apply()
        return jsonify({"applied": count, "status": migrations.status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# Web Search Endpoints (Free - No API Keys)
# ============================================================

@app.route('/api/search', methods=['POST'])
def search_web():
    """Search the web for information."""
    if not web_search:
        return jsonify({"error": "Web search not available"}), 503
    
    data = request.json or {}
    query = data.get('query')
    num_results = data.get('num_results', 5)
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    try:
        results = web_search.search(query, num_results)
        return jsonify({"query": query, "results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search/answer', methods=['POST'])
def search_answer():
    """Get a direct answer to a question."""
    if not web_search:
        return jsonify({"error": "Web search not available"}), 503
    
    data = request.json or {}
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    try:
        answer = web_search.answer(query)
        if answer:
            return jsonify({
                "query": query,
                "answer": {
                    "title": answer.title,
                    "content": answer.content,
                    "source": answer.source,
                    "url": answer.url,
                    "type": answer.type,
                    "confidence": answer.confidence
                }
            })
        else:
            return jsonify({"query": query, "answer": None, "message": "No direct answer found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search/lookup', methods=['POST'])
def search_lookup():
    """Look up a specific topic."""
    if not web_search:
        return jsonify({"error": "Web search not available"}), 503
    
    data = request.json or {}
    topic = data.get('topic')
    
    if not topic:
        return jsonify({"error": "Topic required"}), 400
    
    try:
        info = web_search.lookup(topic)
        return jsonify({"topic": topic, "info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search/facts', methods=['POST'])
def search_facts():
    """Get quick facts about a query."""
    if not web_search:
        return jsonify({"error": "Web search not available"}), 503
    
    data = request.json or {}
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    try:
        facts = web_search.quick_facts(query)
        return jsonify(facts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/wikipedia/<path:title>', methods=['GET'])
def wikipedia_summary(title):
    """Get Wikipedia summary for a topic."""
    if not web_search:
        return jsonify({"error": "Web search not available"}), 503
    
    try:
        summary = web_search.wiki.summary(title)
        return jsonify(summary if summary else {"error": "Article not found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# Serve Frontend (from web/ folder)
# ============================================================

@app.route('/')
def index():
    """Serve the main interface."""
    return send_file('web/aion_interface.html')

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard."""
    return send_file('web/aion_dashboard.html')

@app.route('/ide')
def ide():
    """Serve the IDE."""
    return send_file('web/aion_ide.html')

@app.route('/physics')
def physics():
    """Serve physics visualizer."""
    return send_file('web/physics_visualizer.html')

@app.route('/protein')
def protein():
    """Serve protein folding visualizer."""
    return send_file('web/protein_folding.html')

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    """Serve static assets."""
    return send_file(f'web/assets/{filename}')

# ============================================================
# Run Server
# ============================================================

if __name__ == '__main__':
    print("\n🌐 AION Web Interface running at: http://localhost:5001")
    print("📡 API Documentation: All endpoints accept POST with JSON body")
    print("📂 Frontend routes: /, /dashboard, /ide, /physics, /protein\n")
    app.run(host='0.0.0.0', port=5001, debug=False)
