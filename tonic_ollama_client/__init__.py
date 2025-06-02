"""
Tonic Ollama Client - A robust wrapper for Ollama API.
Assumes Ollama server is managed externally.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union
import socket

# Third-party imports
from ollama import AsyncClient, ChatResponse, ResponseError
from ollama._types import (
    EmbeddingsResponse,
    Message,
)
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Message for server not running
OLLAMA_SERVER_NOT_RUNNING_MESSAGE = """
[bold red]Ollama server is not running or not responsive at {base_url}.[/bold red]

Please ensure the Ollama server is running externally.
You can typically start it with: [bold cyan]ollama serve[/bold cyan]
"""

# Retry configuration for API calls
API_RETRY_CONFIG = {
    "retry": retry_if_exception_type((ConnectionError, TimeoutError, ResponseError)),
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=1, max=10),
    "reraise": True
}

def fancy_print(
    console: Console,
    message: str,
    style: Optional[str] = None,
    panel: bool = False,
    border_style: Optional[str] = None,
):
    """Print formatted messages using Rich."""
    if panel:
        console.print(Panel(message, border_style=border_style or "blue"))
    else:
        console.print(message, style=style)


class ClientConfig(BaseModel):
    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    max_server_startup_attempts: int = Field(default=3, description="Max server responsiveness check attempts")
    debug: bool = Field(default=False, description="Enable debug output")


class OllamaServerNotRunningError(ConnectionError):
    """Error for when Ollama server is not running or unresponsive."""
    def __init__(self, base_url: str = "http://localhost:11434", message: Optional[str] = None):
        self.base_url = base_url
        detail_message = message or OLLAMA_SERVER_NOT_RUNNING_MESSAGE.format(base_url=base_url)
        super().__init__(detail_message)


class TonicOllamaClient:
    """Async client for Ollama API with an externally managed server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        max_server_startup_attempts: int = 3, # For checking responsiveness
        debug: bool = False,
        console: Optional[Console] = None,
    ):
        if console is None:
            console = Console()

        self.config = ClientConfig(
            base_url=base_url,
            max_server_startup_attempts=max_server_startup_attempts,
            debug=debug,
        )
        self.base_url = self.config.base_url
        self.max_server_startup_attempts = self.config.max_server_startup_attempts
        self.debug = self.config.debug
        self.console = console
        self.async_client: Optional[AsyncClient] = None
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        
        if self.debug:
            fancy_print(self.console, f"Initialized TonicOllamaClient (base_url={base_url}, external server).", style="dim blue")

    def get_async_client(self) -> AsyncClient:
        """Get or create the async client instance."""
        if not self.async_client:
            self.async_client = AsyncClient(host=self.base_url)
        return self.async_client

    def _is_ollama_server_running_sync(self) -> bool:
        """Synchronously check if Ollama server is responsive."""
        try:
            host_port = self.base_url.replace("http://", "").replace("https://", "")
            if ":" not in host_port:
                host = host_port
                port = 11434
            else:
                host, port_str = host_port.split(":")
                port = int(port_str)

            with socket.create_connection((host, port), timeout=1):
                if self.debug:
                    fancy_print(self.console, f"Ollama server is responsive at {self.base_url}.", style="dim green")
                return True
        except (socket.timeout, ConnectionRefusedError, OSError, ValueError) as e:
            if self.debug:
                fancy_print(self.console, f"Ollama server not responsive at {self.base_url}. Error: {e}", style="dim yellow")
            return False

    async def ensure_server_ready(self) -> None:
        """
        Ensure externally managed Ollama server is responsive.

        Raises:
            OllamaServerNotRunningError: If server unresponsive after attempts.
        """
        for attempt in range(self.max_server_startup_attempts):
            server_ok = await asyncio.to_thread(self._is_ollama_server_running_sync)
            if server_ok:
                if self.debug:
                    fancy_print(self.console, f"Ollama server at {self.base_url} is responsive.", style="green")
                return
            
            if self.debug:
                fancy_print(self.console, f"Ollama server responsiveness check failed for {self.base_url} (attempt {attempt + 1}/{self.max_server_startup_attempts}). Retrying in 2s...", style="yellow")
            await asyncio.sleep(2)
        
        fancy_print(self.console, OLLAMA_SERVER_NOT_RUNNING_MESSAGE.format(base_url=self.base_url), panel=True, border_style="red")
        raise OllamaServerNotRunningError(self.base_url)

    async def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Create new conversation or return existing if ID provided."""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        return conversation_id

    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get messages in a conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation with ID {conversation_id} does not exist")

        return self.conversations[conversation_id]

    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self.conversations.keys())

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear a conversation's messages."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation with ID {conversation_id} does not exist")

        self.conversations[conversation_id] = []

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation with ID {conversation_id} does not exist")

        del self.conversations[conversation_id]

    @retry(**API_RETRY_CONFIG)
    async def chat(
        self,
        model: str,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Union[Dict[str, Any], ChatResponse]:
        """Send chat message, get response, manage conversation history."""
        await self.ensure_server_ready()
            
        if conversation_id is None:
            conversation_id = await self.create_conversation()
        elif conversation_id not in self.conversations:
            await self.create_conversation(conversation_id)

        messages = self.get_conversation(conversation_id).copy()

        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": system_prompt})

        user_message = {"role": "user", "content": message}
        messages.append(user_message)
        self.conversations[conversation_id].append(user_message)

        try:
            client = self.get_async_client()

            if self.debug:
                fancy_print(self.console, f"Sending chat request to model '{model}'", style="dim blue")

            response = await client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
                stream=False,
            )

            assistant_message = {
                "role": "assistant",
                "content": response["message"]["content"]
            }
            self.conversations[conversation_id].append(assistant_message)

            if self.debug:
                fancy_print(self.console, f"Received response from model '{model}'", style="dim blue")

            return response

        except ResponseError as e:
            fancy_print(self.console, f"Ollama API error: {str(e)}", style="red")
            raise
        except Exception as e:
            fancy_print(self.console, f"Error in chat: {str(e)}", style="red")
            raise

    @retry(**API_RETRY_CONFIG)
    async def generate_embedding(self, model: str, text: str) -> List[float]:
        """Generate embeddings for given text."""
        await self.ensure_server_ready()
            
        try:
            client = self.get_async_client()

            if self.debug:
                fancy_print(self.console, f"Generating embeddings with model '{model}'", style="dim blue")

            response = await client.embeddings(model=model, prompt=text)
            
            if self.debug:
                fancy_print(self.console, f"Generated embeddings with {len(response['embedding'])} dimensions", style="dim blue")
            return response["embedding"]
        except ResponseError as e:
            fancy_print(self.console, f"Ollama API error: {str(e)}", style="red")
            raise
        except Exception as e:
            fancy_print(self.console, f"Error generating embeddings: {str(e)}", style="red")
            raise

def create_client(
    base_url: str = "http://localhost:11434",
    max_server_startup_attempts: int = 3,
    debug: bool = False,
    console: Optional[Console] = None,
) -> TonicOllamaClient:
    """Create a pre-configured TonicOllama client."""
    if console is None:
        console_instance = Console()
    else:
        console_instance = console

    return TonicOllamaClient(
        base_url=base_url,
        max_server_startup_attempts=max_server_startup_attempts,
        debug=debug,
        console=console_instance,
    )

__all__ = [
    "AsyncClient",
    "ChatResponse",
    "ClientConfig",
    "EmbeddingsResponse",
    "Message",
    "OllamaServerNotRunningError",
    "ResponseError",
    "TonicOllamaClient",
    "create_client",
    "fancy_print",
]

