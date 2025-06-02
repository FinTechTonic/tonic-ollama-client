"""
Tonic Ollama Client - A robust wrapper for Ollama API.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union
import socket
import os

# Third-party imports
from ollama import AsyncClient, ChatResponse, ResponseError
from ollama._types import (
    EmbeddingsResponse,
    Message,
    SubscriptableBaseModel,
    Tool,
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

# Message template for model not ready
OLLAMA_MODEL_NOT_READY_MESSAGE = """
[bold yellow]The Ollama model '{model_name}' is not ready.[/bold yellow]
Please ensure:
1. Ollama server is running (run 'ollama serve' in a separate terminal)
2. The model is pulled (run 'ollama pull {model_name}')
"""

# Error message for server not running
OLLAMA_SERVER_NOT_RUNNING_MESSAGE = """
[bold red]Ollama server is not running.[/bold red]

Please start the Ollama server by running the following command in a terminal:
[bold cyan]ollama serve[/bold cyan]

Then try running your command again.
"""

# Define common retry configuration
RETRY_CONFIG = {
    "retry": retry_if_exception_type((ConnectionError, TimeoutError)),
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
    """Print formatted messages using Rich console."""
    if panel:
        console.print(Panel(message, border_style=border_style or "blue"))
    else:
        console.print(message, style=style)


class ClientConfig(BaseModel):
    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    max_readiness_attempts: int = Field(default=3, description="Maximum model readiness check attempts")
    debug: bool = Field(default=False, description="Enable debug output")


class ModelNotReadyError(ResponseError):
    """Error raised when a model cannot be made ready."""
    def __init__(self, model_name: str, reason: str = "Model could not be loaded or made ready"):
        super().__init__(f"{reason}: {model_name}", status_code=400)
        self.model_name = model_name


class OllamaServerNotRunningError(ConnectionError):
    """Error raised when the Ollama server is not running."""
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__(f"Ollama server not running at {base_url}. Please start it with 'ollama serve'")
        self.base_url = base_url


class TonicOllamaClient:
    """Enhanced Ollama client with retries, validation, and conversation management."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        max_readiness_attempts: int = 3,
        debug: bool = False,
        console: Console = Console(),
    ):
        """
        Initialize the TonicOllama client.

        Args:
            base_url: Ollama API base URL.
            max_readiness_attempts: Max attempts for model readiness checks.
            debug: Enable debug output.
            console: Rich Console instance.
        """
        self.config = ClientConfig(
            base_url=base_url,
            max_readiness_attempts=max_readiness_attempts,
            debug=debug
        )
        self.base_url = self.config.base_url
        self.max_readiness_attempts = self.config.max_readiness_attempts
        self.debug = self.config.debug
        self.console = console
        self.async_client: Optional[AsyncClient] = None
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        
        if self.debug:
            fancy_print(self.console, f"Initialized TonicOllamaClient with base_url={base_url}", style="dim blue")

    def get_async_client(self) -> AsyncClient:
        """Get or create the async client instance."""
        if not self.async_client:
            self.async_client = AsyncClient(host=self.base_url)
        return self.async_client

    def _is_ollama_server_running(self) -> bool:
        """Checks if the Ollama server is responsive."""
        try:
            host_port = self.base_url.replace("http://", "").replace("https://", "")
            if ":" not in host_port:  # Default port if not specified
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

    @retry(**RETRY_CONFIG)
    async def check_model_ready(
        self, model_name: str, ready_prompt: str = "Respond with exactly one word: READY"
    ) -> None:
        """
        Check if an Ollama model is ready. Pulls if not found.

        Args:
            model_name: Name of the Ollama model.
            ready_prompt: Prompt to check model readiness.

        Raises:
            OllamaServerNotRunningError: If Ollama server isn't running.
            ModelNotReadyError: If model can't be made ready.
            ResponseError: For API errors.
            ConnectionError: For server connection issues.
        """
        server_running = self._is_ollama_server_running()
        if not server_running:
            fancy_print(self.console, OLLAMA_SERVER_NOT_RUNNING_MESSAGE, panel=True, border_style="red")
            raise OllamaServerNotRunningError(self.base_url)

        consecutive_model_failures = 0
        client = self.get_async_client()
        model_pulled_successfully_this_attempt = False

        while True:
            try:
                if not model_pulled_successfully_this_attempt:
                    fancy_print(self.console, f"Checking local availability of model '{model_name}'...", style="yellow")
                    try:
                        listed_models_response = await client.list()
                        model_tag_to_check = model_name if ":" in model_name else f"{model_name}:latest"
                        found_model = any(m.get('name') == model_tag_to_check for m in listed_models_response.get('models', []))

                        if not found_model:
                            fancy_print(self.console, f"Model '{model_name}' not found locally. Attempting to pull...", style="yellow", panel=True)
                            fancy_print(self.console, "This may take a few minutes depending on the model size and your internet connection.", style="cyan")
                            try:
                                pull_status = await client.pull(model=model_name, stream=False)
                                if pull_status and pull_status.get('status') == 'success':
                                    fancy_print(self.console, f"Model '{model_name}' pulled successfully.", style="green")
                                    model_pulled_successfully_this_attempt = True
                                else:
                                    fancy_print(self.console, f"Failed to pull model '{model_name}'. Status: {pull_status.get('status', 'unknown')}", style="red")
                            except ResponseError as pull_err:
                                fancy_print(self.console, f"Error pulling model '{model_name}': {str(pull_err)}", style="red")
                        else:
                            fancy_print(self.console, f"Model '{model_name}' (as '{model_tag_to_check}') found locally.", style="green")
                            model_pulled_successfully_this_attempt = True

                    except ResponseError as list_err:
                        fancy_print(self.console, f"Ollama API error while listing models: {str(list_err)}", style="red")
                    except Exception as e_list_pull:
                        fancy_print(self.console, f"Unexpected error during model list/pull: {str(e_list_pull)}", style="red")

                if model_pulled_successfully_this_attempt:
                    fancy_print(self.console, f"Checking if Ollama model '{model_name}' is responsive...", style="yellow")
                    response = await client.chat(
                        model=model_name,
                        messages=[{"role": "user", "content": ready_prompt}],
                        options={"temperature": 0.1},
                        stream=False,
                    )
                    raw_content = response.get("message", {}).get("content", "").strip()
                    processed_content = raw_content.upper()
                    if processed_content.endswith((".", "!", "?")):
                        processed_content = processed_content[:-1]

                    if processed_content == "READY":
                        fancy_print(self.console, f"Ollama model '{model_name}' is responsive.", style="green")
                        return
                    else:
                        fancy_print(self.console, f"Ollama model responded, but not with expected 'READY'. Response: {raw_content}", style="yellow")
                
                consecutive_model_failures += 1
                model_pulled_successfully_this_attempt = False

            except ResponseError as e:
                fancy_print(self.console, f"Ollama API error during readiness check: {str(e)}", style="red")
                consecutive_model_failures += 1
                model_pulled_successfully_this_attempt = False
            except ConnectionError as e_conn:
                fancy_print(self.console, f"Connection lost to Ollama server during model readiness: {e_conn}", style="red")
                fancy_print(self.console, OLLAMA_SERVER_NOT_RUNNING_MESSAGE, panel=True, border_style="red")
                raise OllamaServerNotRunningError(self.base_url)
            except Exception as e_other:
                fancy_print(self.console, f"Unexpected error checking model readiness: {str(e_other)}", style="red")
                consecutive_model_failures += 1
                model_pulled_successfully_this_attempt = False

            if consecutive_model_failures >= self.max_readiness_attempts:
                fancy_print(self.console, f"Max model readiness attempts ({self.max_readiness_attempts}) reached for '{model_name}'.", style="red")
                raise ModelNotReadyError(model_name, f"Failed to make model '{model_name}' ready after {self.max_readiness_attempts} attempts.")

            fancy_print(self.console, f"Retrying model readiness for '{model_name}' (attempt {consecutive_model_failures + 1}/{self.max_readiness_attempts})...", style="cyan")
            await asyncio.sleep(2)

    async def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Create a new conversation or return existing if ID provided."""
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

    @retry(**RETRY_CONFIG)
    async def chat(
        self,
        model: str,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Union[Dict[str, Any], ChatResponse]:
        """Send a chat message and get a response, managing conversation history."""
        server_running = self._is_ollama_server_running()
        if not server_running:
            fancy_print(self.console, OLLAMA_SERVER_NOT_RUNNING_MESSAGE, panel=True, border_style="red")
            raise OllamaServerNotRunningError(self.base_url)
            
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

    @retry(**RETRY_CONFIG)
    async def generate_embedding(self, model: str, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        server_running = self._is_ollama_server_running()
        if not server_running:
            fancy_print(self.console, OLLAMA_SERVER_NOT_RUNNING_MESSAGE, panel=True, border_style="red")
            raise OllamaServerNotRunningError(self.base_url)
            
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
    max_readiness_attempts: int = 3,
    debug: bool = False,
    console: Optional[Console] = None,
) -> TonicOllamaClient:
    """Create a pre-configured TonicOllama client instance."""
    if console is None:
        console_instance = Console()
    else:
        console_instance = console

    return TonicOllamaClient(
        base_url=base_url,
        max_readiness_attempts=max_readiness_attempts,
        debug=debug,
        console=console_instance,
    )

__all__ = [
    "AsyncClient",
    "ChatResponse",
    "ClientConfig",
    "EmbeddingsResponse",
    "Message",
    "ModelNotReadyError",
    "OllamaServerNotRunningError",
    "ResponseError",
    "SubscriptableBaseModel",
    "Tool",
    "TonicOllamaClient",
    "create_client",
    "fancy_print",
]

