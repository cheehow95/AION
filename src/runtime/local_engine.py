"""
AION Local Reasoning Engine
Fast, rule-based reasoning without external API dependencies.
Provides instant inference for agents.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from datetime import datetime
import re
import hashlib


class LocalReasoningEngine:
    """
    Ultra-fast local reasoning engine for AION agents.
    Uses pattern matching, rule-based inference, and cached computations.
    No external API calls required.
    """
    
    def __init__(self):
        self.rules: list[tuple[Callable, Callable]] = []
        self.knowledge_base: dict[str, Any] = {}
        self.inference_cache: dict[str, Any] = {}
        self.reasoning_trace: list[dict] = []
        
        # Load default rules
        self._load_default_rules()
    
    def _cache_key(self, operation: str, *args) -> str:
        """Generate cache key for memoization."""
        content = f"{operation}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _load_default_rules(self):
        """Load built-in reasoning rules."""
        
        # Math expressions
        self.add_rule(
            lambda x: re.match(r'^[\d\s+\-*/().]+$', str(x)),
            lambda x: self._eval_math(x)
        )
        
        # Yes/No questions
        self.add_rule(
            lambda x: str(x).lower().startswith(('is ', 'are ', 'can ', 'do ', 'does ', 'will ', 'should ')),
            lambda x: self._answer_yes_no(x)
        )
        
        # What questions
        self.add_rule(
            lambda x: str(x).lower().startswith('what '),
            lambda x: self._answer_what(x)
        )
        
        # How questions
        self.add_rule(
            lambda x: str(x).lower().startswith('how '),
            lambda x: self._answer_how(x)
        )
        
        # Greeting patterns
        self.add_rule(
            lambda x: any(g in str(x).lower() for g in ['hello', 'hi', 'hey', 'greetings']),
            lambda x: "Hello! I'm an AION agent. How can I help you today?"
        )
        
        # Help requests
        self.add_rule(
            lambda x: 'help' in str(x).lower(),
            lambda x: self._provide_help(x)
        )
    
    def add_rule(self, condition: Callable, action: Callable):
        """Add a reasoning rule."""
        self.rules.append((condition, action))
    
    def add_knowledge(self, key: str, value: Any):
        """Add to knowledge base."""
        self.knowledge_base[key] = value
    
    def think(self, prompt: str = None, context: dict = None) -> str:
        """
        Fast thinking operation - pattern match and reason.
        """
        context = context or {}
        cache_key = self._cache_key('think', prompt, str(context))
        
        if cache_key in self.inference_cache:
            return self.inference_cache[cache_key]
        
        result = self._generate_thought(prompt, context)
        
        self.reasoning_trace.append({
            'type': 'think',
            'input': prompt,
            'output': result,
            'timestamp': datetime.now().isoformat()
        })
        
        self.inference_cache[cache_key] = result
        return result
    
    def analyze(self, target: Any, context: dict = None) -> dict:
        """
        Fast analysis - extract key information and patterns.
        """
        target_str = str(target)
        
        analysis = {
            'type': self._detect_type(target_str),
            'length': len(target_str),
            'keywords': self._extract_keywords(target_str),
            'sentiment': self._detect_sentiment(target_str),
            'intent': self._detect_intent(target_str),
            'entities': self._extract_entities(target_str),
            'complexity': self._assess_complexity(target_str),
        }
        
        self.reasoning_trace.append({
            'type': 'analyze',
            'input': target_str[:100],
            'output': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        return analysis
    
    def reflect(self, target: Any = None, context: dict = None) -> str:
        """
        Fast reflection - review and learn.
        """
        if target:
            reflection = f"Reflecting on: {target}\n"
            reflection += f"Context has {len(context or {})} items.\n"
            reflection += f"Reasoning trace has {len(self.reasoning_trace)} steps.\n"
            
            if self.reasoning_trace:
                last_step = self.reasoning_trace[-1]
                reflection += f"Last action was '{last_step['type']}'"
        else:
            reflection = "General reflection on current state.\n"
            reflection += f"Knowledge base contains {len(self.knowledge_base)} facts.\n"
            reflection += f"Cache has {len(self.inference_cache)} entries."
        
        self.reasoning_trace.append({
            'type': 'reflect',
            'input': str(target)[:50] if target else None,
            'output': reflection,
            'timestamp': datetime.now().isoformat()
        })
        
        return reflection
    
    def decide(self, options: Any, context: dict = None) -> dict:
        """
        Fast decision making - evaluate and choose.
        """
        options_str = str(options)
        
        # Apply rules to find best match
        for condition, action in self.rules:
            try:
                if condition(options_str):
                    result = action(options_str)
                    decision = {
                        'decision': result,
                        'confidence': 0.85,
                        'reasoning': f"Applied rule matching for: {options_str[:50]}",
                        'method': 'rule_based'
                    }
                    
                    self.reasoning_trace.append({
                        'type': 'decide',
                        'input': options_str[:100],
                        'output': decision,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    return decision
            except:
                continue
        
        # Default decision
        decision = {
            'decision': self._default_response(options_str),
            'confidence': 0.5,
            'reasoning': "Used default response generation",
            'method': 'default'
        }
        
        self.reasoning_trace.append({
            'type': 'decide',
            'input': options_str[:100],
            'output': decision,
            'timestamp': datetime.now().isoformat()
        })
        
        return decision
    
    # ========== Helper Methods ==========
    
    def _generate_thought(self, prompt: str, context: dict) -> str:
        """Generate a thought based on prompt and context."""
        if not prompt:
            return "Processing current context and evaluating options..."
        
        # Check if it's about the context
        if context:
            return f"Considering the context with {len(context)} elements. Key aspects: {list(context.keys())[:3]}"
        
        return f"Thinking about: {prompt[:100]}. Analyzing patterns and formulating approach."
    
    def _detect_type(self, text: str) -> str:
        """Detect the type of input."""
        if re.match(r'^[\d\s+\-*/().]+$', text):
            return 'math_expression'
        elif '?' in text:
            return 'question'
        elif any(cmd in text.lower() for cmd in ['please', 'can you', 'could you', 'would you']):
            return 'request'
        elif text.isupper():
            return 'emphasis'
        else:
            return 'statement'
    
    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your',
                      'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they',
                      'them', 'their', 'what', 'which', 'who', 'whom', 'this',
                      'that', 'these', 'those', 'am', 'and', 'but', 'if', 'or'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Return unique keywords, maintaining order
        seen = set()
        return [w for w in keywords if not (w in seen or seen.add(w))][:10]
    
    def _detect_sentiment(self, text: str) -> str:
        """Simple sentiment detection."""
        positive = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                    'love', 'like', 'happy', 'joy', 'thanks', 'thank', 'please', 'awesome']
        negative = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'angry', 'sad',
                    'wrong', 'error', 'fail', 'problem', 'issue', 'bug', 'broken']
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'
    
    def _detect_intent(self, text: str) -> str:
        """Detect user intent."""
        text_lower = text.lower()
        
        if any(q in text_lower for q in ['?', 'what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        elif any(c in text_lower for c in ['calculate', 'compute', 'solve', 'math', '+', '-', '*', '/']):
            return 'calculation'
        elif any(g in text_lower for g in ['hello', 'hi', 'hey', 'greetings', 'good morning']):
            return 'greeting'
        elif any(h in text_lower for h in ['help', 'assist', 'support', 'guide']):
            return 'help_request'
        elif any(s in text_lower for s in ['search', 'find', 'look for', 'locate']):
            return 'search'
        elif any(c in text_lower for c in ['create', 'make', 'build', 'generate', 'write']):
            return 'creation'
        else:
            return 'general'
    
    def _extract_entities(self, text: str) -> dict:
        """Extract named entities."""
        entities = {
            'numbers': re.findall(r'\b\d+(?:\.\d+)?\b', text),
            'emails': re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text),
            'urls': re.findall(r'https?://\S+', text),
            'dates': re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
        }
        return {k: v for k, v in entities.items() if v}
    
    def _assess_complexity(self, text: str) -> str:
        """Assess input complexity."""
        words = text.split()
        if len(words) < 5:
            return 'simple'
        elif len(words) < 20:
            return 'moderate'
        else:
            return 'complex'
    
    def _eval_math(self, expression: str) -> str:
        """Safely evaluate math expressions."""
        try:
            # Only allow safe math operations
            allowed = set('0123456789+-*/.(). ')
            if all(c in allowed for c in expression):
                result = eval(expression)
                return f"The result is: {result}"
        except:
            pass
        return "I couldn't calculate that expression."
    
    def _answer_yes_no(self, question: str) -> str:
        """Answer yes/no questions."""
        q = question.lower()
        
        # Check knowledge base
        for key, value in self.knowledge_base.items():
            if key.lower() in q:
                return f"Based on my knowledge: {value}"
        
        # Definite answers
        if 'aion' in q and 'programming language' in q:
            return "Yes, AION is a declarative AI-native programming language."
        if 'agent' in q:
            return "Yes, AION is designed specifically for building AI agents."
        
        return "I would need more context to answer that definitively."
    
    def _answer_what(self, question: str) -> str:
        """Answer 'what' questions."""
        q = question.lower()
        
        if 'aion' in q:
            return "AION (Artificial Intelligence Oriented Notation) is a declarative programming language for building thinking systems and AI agents."
        if 'agent' in q:
            return "An agent in AION is an autonomous reasoning unit with goals, memory, and capabilities."
        if 'time' in q:
            return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
        if 'date' in q:
            return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
        
        keywords = self._extract_keywords(q)
        if keywords:
            return f"That's a question about: {', '.join(keywords[:3])}. Let me analyze it further."
        
        return "Could you provide more details about what you'd like to know?"
    
    def _answer_how(self, question: str) -> str:
        """Answer 'how' questions."""
        q = question.lower()
        
        if 'agent' in q and 'create' in q:
            return "To create an agent in AION, use: agent AgentName { goal \"...\"; memory working; on input(x): ... }"
        if 'work' in q and 'aion' in q:
            return "AION works by declaring agents with goals, memory, and event handlers. The runtime executes reasoning blocks like think, analyze, reflect, and decide."
        
        return "I can provide guidance on that. What specific aspect would you like to understand?"
    
    def _provide_help(self, text: str) -> str:
        """Provide help response."""
        return """I can help you with:
• Writing AION code
• Understanding agent concepts
• Calculations and computations
• Answering questions
• General assistance

Just ask me anything!"""
    
    def _default_response(self, text: str) -> str:
        """Generate a default response."""
        intent = self._detect_intent(text)
        keywords = self._extract_keywords(text)
        
        if intent == 'greeting':
            return "Hello! How can I assist you today?"
        elif intent == 'question':
            return f"That's an interesting question about {keywords[0] if keywords else 'that topic'}. Let me think about it."
        elif intent == 'calculation':
            return "I can help with calculations. Please provide the expression."
        elif intent == 'help_request':
            return self._provide_help(text)
        else:
            return f"I understand you're interested in {keywords[0] if keywords else 'something'}. How can I help further?"
    
    def get_trace(self) -> list[dict]:
        """Get the reasoning trace."""
        return self.reasoning_trace.copy()
    
    def clear_trace(self):
        """Clear the reasoning trace."""
        self.reasoning_trace.clear()
    
    def clear_cache(self):
        """Clear the inference cache."""
        self.inference_cache.clear()
