"""
AION Model Interface
Provides abstraction layer for LLM integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, AsyncIterator
import asyncio


class ModelError(Exception):
    """Raised when a model operation fails."""
    pass


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # system, user, assistant
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Result from a model completion."""
    content: str
    model: str
    tokens_used: int = 0
    finish_reason: str = "stop"
    metadata: dict = field(default_factory=dict)


class BaseModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        **kwargs
    ) -> CompletionResult:
        """Generate a completion from messages."""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a completion token by token."""
        pass


class OpenAIProvider(BaseModelProvider):
    """OpenAI API provider (GPT-3.5, GPT-4, etc.)."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.model = self.config.get('model', 'gpt-4')
        self.base_url = self.config.get('base_url', 'https://api.openai.com/v1')
    
    async def complete(
        self,
        messages: list[Message],
        **kwargs
    ) -> CompletionResult:
        """Generate completion using OpenAI API."""
        try:
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            formatted_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            
            response = await client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=formatted_messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2048),
            )
            
            return CompletionResult(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason
            )
        except ImportError:
            raise ModelError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise ModelError(f"OpenAI API error: {e}")
    
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion from OpenAI."""
        try:
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            formatted_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            
            stream = await client.chat.completions.create(
                model=kwargs.get('model', self.model),
                messages=formatted_messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2048),
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except ImportError:
            raise ModelError("OpenAI package not installed")
        except Exception as e:
            raise ModelError(f"OpenAI streaming error: {e}")


class AnthropicProvider(BaseModelProvider):
    """Anthropic API provider (Claude)."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.api_key = self.config.get('api_key')
        self.model = self.config.get('model', 'claude-3-sonnet-20240229')
    
    async def complete(
        self,
        messages: list[Message],
        **kwargs
    ) -> CompletionResult:
        """Generate completion using Anthropic API."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            # Separate system message
            system_message = None
            chat_messages = []
            
            for m in messages:
                if m.role == 'system':
                    system_message = m.content
                else:
                    chat_messages.append({
                        "role": m.role,
                        "content": m.content
                    })
            
            response = await client.messages.create(
                model=kwargs.get('model', self.model),
                system=system_message,
                messages=chat_messages,
                max_tokens=kwargs.get('max_tokens', 2048),
            )
            
            return CompletionResult(
                content=response.content[0].text,
                model=response.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason
            )
        except ImportError:
            raise ModelError("Anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise ModelError(f"Anthropic API error: {e}")
    
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion from Anthropic."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            system_message = None
            chat_messages = []
            
            for m in messages:
                if m.role == 'system':
                    system_message = m.content
                else:
                    chat_messages.append({
                        "role": m.role,
                        "content": m.content
                    })
            
            async with client.messages.stream(
                model=kwargs.get('model', self.model),
                system=system_message,
                messages=chat_messages,
                max_tokens=kwargs.get('max_tokens', 2048),
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except ImportError:
            raise ModelError("Anthropic package not installed")
        except Exception as e:
            raise ModelError(f"Anthropic streaming error: {e}")


class OllamaProvider(BaseModelProvider):
    """Ollama local model provider."""
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model = self.config.get('model', 'llama3')
        self.base_url = self.config.get('base_url', 'http://localhost:11434')
    
    async def complete(
        self,
        messages: list[Message],
        **kwargs
    ) -> CompletionResult:
        """Generate completion using Ollama."""
        try:
            import httpx
            
            formatted_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": kwargs.get('model', self.model),
                        "messages": formatted_messages,
                        "stream": False
                    },
                    timeout=120.0
                )
                response.raise_for_status()
                data = response.json()
            
            return CompletionResult(
                content=data['message']['content'],
                model=data.get('model', self.model),
                tokens_used=data.get('eval_count', 0),
                finish_reason="stop"
            )
        except ImportError:
            raise ModelError("httpx package not installed. Run: pip install httpx")
        except Exception as e:
            raise ModelError(f"Ollama API error: {e}")
    
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion from Ollama."""
        try:
            import httpx
            import json
            
            formatted_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": kwargs.get('model', self.model),
                        "messages": formatted_messages,
                        "stream": True
                    },
                    timeout=120.0
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if 'message' in data and 'content' in data['message']:
                                yield data['message']['content']
        except ImportError:
            raise ModelError("httpx package not installed")
        except Exception as e:
            raise ModelError(f"Ollama streaming error: {e}")


class ModelRegistry:
    """Registry for model providers and instances."""
    
    _providers = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'ollama': OllamaProvider,
        'local': OllamaProvider,  # Alias
    }
    
    _instances: dict[str, BaseModelProvider] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a new provider class."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_model(
        cls,
        name: str,
        provider: str = 'openai',
        config: dict = None
    ) -> BaseModelProvider:
        """Create and register a model instance."""
        if provider not in cls._providers:
            raise ModelError(f"Unknown provider: {provider}")
        
        instance = cls._providers[provider](config)
        cls._instances[name] = instance
        return instance
    
    @classmethod
    def get_model(cls, name: str) -> BaseModelProvider:
        """Get a registered model instance."""
        if name not in cls._instances:
            raise ModelError(f"Model not registered: {name}")
        return cls._instances[name]
