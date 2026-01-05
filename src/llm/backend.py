"""
AION LLM Backend Integration
============================

Connect AION to top LLM providers for state-of-the-art text generation.
Supports: OpenAI (GPT-5), Anthropic (Claude), Google (Gemini), Local models

Usage:
    from aion import AION
    ai = AION()
    
    # Configure LLM backend
    ai.llm.configure("openai", api_key="...")
    
    # Generate with top models
    response = ai.llm.generate("Write a poem about AI")
    
    # Use specific model
    response = ai.llm.chat("gpt-5.2-high", messages=[...])
"""

import os
import json
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    AUTO = "auto"


@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    base_url: Optional[str] = None


class LLMBackend:
    """Unified LLM backend for AION - connects to top models."""
    
    # Model rankings based on LM Arena
    MODEL_RANKINGS = {
        # Text (higher score = better)
        "gemini-3-pro": {"score": 1490, "provider": "google"},
        "gemini-3-flash": {"score": 1480, "provider": "google"},
        "grok-4.1-thinking": {"score": 1477, "provider": "xai"},
        "claude-opus-4-5-thinking": {"score": 1470, "provider": "anthropic"},
        "gpt-5.1-high": {"score": 1458, "provider": "openai"},
        "gpt-5.2": {"score": 1394, "provider": "openai"},
        "claude-sonnet-4-5": {"score": 1450, "provider": "anthropic"},
        
        # Aliases for easy access
        "best": {"score": 1490, "provider": "google", "model": "gemini-3-pro"},
        "fast": {"score": 1480, "provider": "google", "model": "gemini-3-flash"},
        "thinking": {"score": 1477, "provider": "anthropic", "model": "claude-opus-4-5-thinking"},
    }
    
    def __init__(self):
        self._configs: Dict[str, LLMConfig] = {}
        self._current_provider = LLMProvider.AUTO
        self._clients = {}
        self._cache = {}
        self._init_from_env()
    
    def _init_from_env(self):
        """Initialize from environment variables."""
        if os.getenv("OPENAI_API_KEY"):
            self.configure("openai", os.getenv("OPENAI_API_KEY"))
        if os.getenv("ANTHROPIC_API_KEY"):
            self.configure("anthropic", os.getenv("ANTHROPIC_API_KEY"))
        if os.getenv("GOOGLE_API_KEY"):
            self.configure("google", os.getenv("GOOGLE_API_KEY"))
    
    def configure(self, provider: str, api_key: str, model: str = None, **kwargs):
        """Configure an LLM provider."""
        provider_enum = LLMProvider(provider)
        
        default_models = {
            LLMProvider.OPENAI: "gpt-5.2",
            LLMProvider.ANTHROPIC: "claude-opus-4-5",
            LLMProvider.GOOGLE: "gemini-3-pro",
            LLMProvider.LOCAL: "llama-3-70b",
        }
        
        self._configs[provider] = LLMConfig(
            provider=provider_enum,
            api_key=api_key,
            model=model or default_models.get(provider_enum, "gpt-5.2"),
            **kwargs
        )
        
        # Initialize client
        self._init_client(provider)
    
    def _init_client(self, provider: str):
        """Initialize provider client."""
        config = self._configs.get(provider)
        if not config:
            return
        
        try:
            if config.provider == LLMProvider.OPENAI:
                from openai import OpenAI
                self._clients["openai"] = OpenAI(api_key=config.api_key)
            elif config.provider == LLMProvider.ANTHROPIC:
                from anthropic import Anthropic
                self._clients["anthropic"] = Anthropic(api_key=config.api_key)
            elif config.provider == LLMProvider.GOOGLE:
                import google.generativeai as genai
                genai.configure(api_key=config.api_key)
                self._clients["google"] = genai
        except ImportError as e:
            print(f"Warning: {provider} client not installed: {e}")
    
    def generate(self, prompt: str, model: str = "best", **kwargs) -> str:
        """Generate text using the best available model."""
        # Resolve model alias
        model_info = self.MODEL_RANKINGS.get(model, {})
        provider = model_info.get("provider", "openai")
        actual_model = model_info.get("model", model)
        
        # Try providers in order of ranking
        if provider == "google" and "google" in self._clients:
            return self._generate_google(prompt, actual_model, **kwargs)
        elif provider == "anthropic" and "anthropic" in self._clients:
            return self._generate_anthropic(prompt, actual_model, **kwargs)
        elif provider == "openai" and "openai" in self._clients:
            return self._generate_openai(prompt, actual_model, **kwargs)
        
        # Fallback to any available
        if "openai" in self._clients:
            return self._generate_openai(prompt, "gpt-5.2", **kwargs)
        if "anthropic" in self._clients:
            return self._generate_anthropic(prompt, "claude-opus-4-5", **kwargs)
        if "google" in self._clients:
            return self._generate_google(prompt, "gemini-3-pro", **kwargs)
        
        return self._generate_local(prompt, **kwargs)
    
    def _generate_openai(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using OpenAI."""
        client = self._clients.get("openai")
        if not client:
            return "OpenAI not configured"
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using Anthropic."""
        client = self._clients.get("anthropic")
        if not client:
            return "Anthropic not configured"
        
        response = client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _generate_google(self, prompt: str, model: str, **kwargs) -> str:
        """Generate using Google Gemini."""
        client = self._clients.get("google")
        if not client:
            return "Google not configured"
        
        gen_model = client.GenerativeModel(model)
        response = gen_model.generate_content(prompt)
        return response.text
    
    def _generate_local(self, prompt: str, **kwargs) -> str:
        """Fallback to local generation."""
        # Use AION's built-in capabilities
        return f"[Local] Processing: {prompt[:100]}..."
    
    def chat(self, messages: List[Dict], model: str = "best", **kwargs) -> str:
        """Multi-turn chat with context."""
        model_info = self.MODEL_RANKINGS.get(model, {})
        provider = model_info.get("provider", "openai")
        
        if provider == "openai" and "openai" in self._clients:
            response = self._clients["openai"].chat.completions.create(
                model=model_info.get("model", model),
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        
        # Convert to single prompt for other providers
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return self.generate(prompt, model, **kwargs)
    
    def stream(self, prompt: str, model: str = "best") -> Generator[str, None, None]:
        """Stream response tokens."""
        if "openai" in self._clients:
            response = self._clients["openai"].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            yield self.generate(prompt, model)
    
    def with_thinking(self, prompt: str, model: str = "thinking") -> Dict:
        """Generate with extended thinking (like o1/claude-thinking)."""
        thinking_prompt = f"""Think through this step by step:

{prompt}

First, break down the problem. Then analyze each part. Finally, synthesize your answer.
Show your reasoning process."""
        
        response = self.generate(thinking_prompt, model)
        
        return {
            "prompt": prompt,
            "model": model,
            "thinking": True,
            "response": response
        }
    
    def list_models(self) -> List[Dict]:
        """List available models with rankings."""
        models = []
        for name, info in self.MODEL_RANKINGS.items():
            if not name.startswith("_"):
                models.append({
                    "name": name,
                    "score": info.get("score", 0),
                    "provider": info.get("provider", "unknown"),
                    "available": info.get("provider") in self._clients
                })
        return sorted(models, key=lambda x: x["score"], reverse=True)
    
    def get_best_model(self) -> str:
        """Get the best available model."""
        for model in self.list_models():
            if model["available"]:
                return model["name"]
        return "local"


# Integration with AION
class LLMDomain:
    """LLM integration domain for AION."""
    
    def __init__(self):
        self._backend = LLMBackend()
    
    def configure(self, provider: str, api_key: str, **kwargs):
        """Configure an LLM provider."""
        self._backend.configure(provider, api_key, **kwargs)
    
    def generate(self, prompt: str, model: str = "best", **kwargs) -> str:
        """Generate text using top models."""
        return self._backend.generate(prompt, model, **kwargs)
    
    def chat(self, messages: List[Dict], model: str = "best") -> str:
        """Multi-turn conversation."""
        return self._backend.chat(messages, model)
    
    def stream(self, prompt: str, model: str = "best"):
        """Stream response."""
        return self._backend.stream(prompt, model)
    
    def think(self, prompt: str) -> Dict:
        """Extended thinking mode."""
        return self._backend.with_thinking(prompt)
    
    def list_models(self) -> List[Dict]:
        """List available models."""
        return self._backend.list_models()
    
    def best_model(self) -> str:
        """Get best available model."""
        return self._backend.get_best_model()
