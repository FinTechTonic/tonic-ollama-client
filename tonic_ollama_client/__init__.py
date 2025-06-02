"""
Tonic Ollama Client - A robust wrapper for Ollama API.
Assumes Ollama server is managed externally.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, TypeVar, overload, Literal, cast
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

# Constants
CONCURRENT_MODELS = 1  # Only one model can be loaded by default
DEFAULT_MODELS_TO_UNLOAD_ON_CLOSE = ["llama3.1:latest", "phi4:latest", "qwen3:8b", "mistral:latest"] # Models to attempt to unload
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
    concurrent_models: int = Field(default=CONCURRENT_MODELS, description="Maximum number of concurrent models (always 1)")
    stream_responses: bool = Field(default=False, description="Stream responses from chat by default")


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
        max_server_startup_attempts: int = 3,
        debug: bool = False,
        console: Optional[Console] = None,
        concurrent_models: int = CONCURRENT_MODELS,
        models_to_unload_on_close: Optional[List[str]] = None,
        stream_responses: bool = False,
    ):
        if console is None:
            console = Console()

        # Always enforce CONCURRENT_MODELS=1 regardless of what was passed
        if concurrent_models != CONCURRENT_MODELS:
            if debug:
                fancy_print(console, 
                    f"Warning: concurrent_models={concurrent_models} was requested but only {CONCURRENT_MODELS} is supported. Using {CONCURRENT_MODELS}.", 
                    style="yellow")
            # concurrent_models = CONCURRENT_MODELS # This line was here, but self.concurrent_models below uses the class attribute

        self.config = ClientConfig(
            base_url=base_url,
            max_server_startup_attempts=max_server_startup_attempts,
            debug=debug,
            concurrent_models=CONCURRENT_MODELS,  # Always use the constant
            stream_responses=stream_responses, # Store new parameter
        )
        self.base_url = self.config.base_url
        self.max_server_startup_attempts = self.config.max_server_startup_attempts
        self.debug = self.config.debug
        self.concurrent_models = self.config.concurrent_models # This should be CONCURRENT_MODELS
        self.stream_responses = self.config.stream_responses # New attribute
        self.console = console
        self.async_client: Optional[AsyncClient] = None
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        
        self._model_semaphore = asyncio.Semaphore(CONCURRENT_MODELS) # Use the enforced CONCURRENT_MODELS
        
        self.models_to_unload_on_close = models_to_unload_on_close if models_to_unload_on_close is not None else DEFAULT_MODELS_TO_UNLOAD_ON_CLOSE.copy()

        if self.debug:
            fancy_print(self.console, 
                f"Initialized TonicOllamaClient (base_url={base_url}, concurrent_models={self.concurrent_models}, stream_default={self.stream_responses}, default_unload_list_size={len(self.models_to_unload_on_close)})", 
                style="dim blue")

    def get_async_client(self) -> AsyncClient:
        """Get or create the async client instance."""
        if not self.async_client:
            self.async_client = AsyncClient(host=self.base_url)
        assert self.async_client is not None, "Async client should be initialized."
        return self.async_client
    
    def get_available_model_slots(self) -> int:
        """Return the number of available slots for concurrent model usage."""
        return self._model_semaphore._value

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

    @overload
    async def chat(
        self,
        model: str,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stream: Literal[False] = False,
    ) -> Dict[str, Any]:
        ...
    
    @overload
    async def chat(
        self,
        model: str,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stream: Literal[True] = True,
    ) -> AsyncGenerator[ChatResponse, None]:
        ...

    @retry(**API_RETRY_CONFIG)
    async def chat(
        self,
        model: str,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stream: Optional[bool] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[ChatResponse, None]]:
        """Send chat message, get response, manage conversation history. Can stream responses."""
        await self.ensure_server_ready()

        if stream is None:
            stream = self.stream_responses
        
        # Determine if the 'think' parameter should be passed to the ollama client
        use_ollama_think_param = "qwen3" in model.lower()

        # First prepare the conversation outside the semaphore
        if conversation_id is None:
            conversation_id = await self.create_conversation()
        elif conversation_id not in self.conversations:
            await self.create_conversation(conversation_id)

        messages = self.get_conversation(conversation_id).copy()

        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": system_prompt})

        user_message = {"role": "user", "content": message}
        messages.append(user_message)
        # For non-streaming, add user message to history now. For streaming, it's already added.
        # The self.conversations[conversation_id] is updated with user message *before* the call.
        # Assistant message is added *after* for non-streaming, or *after full accumulation* for streaming.
        if not stream: # Only add user message to persistent history if not streaming here, as it's done before the loop for streaming
             if not any(msg['role'] == 'user' and msg['content'] == message for msg in self.conversations[conversation_id]):
                self.conversations[conversation_id].append(user_message)
        else: # For streaming, ensure user message is in history before starting
            self.conversations[conversation_id].append(user_message)


        # Use semaphore to limit concurrent model access
        async with self._model_semaphore:
            if self.debug:
                fancy_print(self.console, 
                    f"Acquired model semaphore for chat with '{model}' ({self.get_available_model_slots()}/{self.concurrent_models} slots available, stream={stream})", 
                    style="dim blue")
            
            try:
                client = self.get_async_client()

                if self.debug:
                    fancy_print(self.console, f"Sending chat request to model '{model}' (stream={stream}, ollama_think={use_ollama_think_param})", style="dim blue")

                if stream:
                    # Handle streaming response
                    response_stream = await client.chat(
                        model=model,
                        messages=messages, # Send the prepared messages list
                        options={"temperature": temperature},
                        stream=True,
                        think=use_ollama_think_param, # Pass think parameter based on model
                    )
                    
                    # Define the async generator within the semaphore context
                    async def stream_generator() -> AsyncGenerator[ChatResponse, None]:
                        full_assistant_content = ""
                        try:
                            async for part in response_stream: # part is already ChatResponse
                                yield part # Removed redundant cast
                                if hasattr(part, 'message') and hasattr(part.message, 'content') and part.message.content:
                                    full_assistant_content += part.message.content
                                if hasattr(part, 'done') and part.done:
                                    assistant_message_for_history = {
                                        "role": "assistant",
                                        "content": full_assistant_content
                                    }
                                    self.conversations[conversation_id].append(assistant_message_for_history)
                                    if self.debug:
                                        fancy_print(self.console, f"Stream for '{model}' completed. Full response added to history.", style="dim blue")
                        finally:
                            if self.debug:
                                fancy_print(self.console, 
                                    f"Released model semaphore for chat stream with '{model}'", 
                                    style="dim blue")
                    return stream_generator()

                else:
                    # Handle non-streaming response
                    response: ChatResponse = await client.chat(
                        model=model,
                        messages=messages, # Send the prepared messages list
                        options={"temperature": temperature},
                        stream=False,
                        think=use_ollama_think_param, # Pass think parameter based on model
                    )

                    assistant_message_content = response.message.content
                    # Include thinking if present and not empty (this happens if use_ollama_think_param was True)
                    if use_ollama_think_param and response.message.thinking:
                        assistant_message_content = f"<thinking>{response.message.thinking}</thinking>\n{assistant_message_content}"


                    assistant_message = {
                        "role": "assistant",
                        "content": assistant_message_content
                    }
                    self.conversations[conversation_id].append(assistant_message)

                    if self.debug:
                        fancy_print(self.console, f"Received response from model '{model}'", style="dim blue")
                    
                    dumped_response = response.model_dump()
                    # If ollama_think was used and thinking content exists, ensure it's in the dumped response
                    if use_ollama_think_param and response.message.thinking:
                        if "message" in dumped_response:
                            # The content already includes thinking if it was prepended
                            dumped_response["message"]["content"] = assistant_message_content
                            # Also add a separate 'thinking' field for clarity if it existed
                            dumped_response["message"]["thinking"] = response.message.thinking
                    return dumped_response

            except ResponseError as e:
                fancy_print(self.console, f"Ollama API error: {str(e)}", style="red")
                raise
            except Exception as e:
                fancy_print(self.console, f"Error in chat: {str(e)}", style="red")
                raise
            finally:
                if not stream: # Only release here if not streaming, stream_generator handles its own release
                    if self.debug:
                        fancy_print(self.console, 
                            f"Released model semaphore for chat with '{model}'", 
                            style="dim blue")

    @retry(**API_RETRY_CONFIG)
    async def generate_embedding(self, model: str, text: str) -> List[float]:
        """Generate embeddings for given text."""
        await self.ensure_server_ready()
        
        # Use semaphore to limit concurrent model access
        async with self._model_semaphore:
            if self.debug:
                fancy_print(self.console, 
                    f"Acquired model semaphore for embeddings with '{model}' ({self.get_available_model_slots()}/{self.concurrent_models} slots available)", 
                    style="dim blue")
            
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
            finally:
                if self.debug:
                    fancy_print(self.console, 
                        f"Released model semaphore for embeddings with '{model}'", 
                        style="dim blue")

    async def close(self, model_to_unload: Optional[str] = None):
        """
        Attempt to unload specified model(s) and close the underlying HTTP client.
        If model_to_unload is specified, only that model is targeted.
        Otherwise, models from self.models_to_unload_on_close are targeted.
        This is a best-effort operation.
        """
        if self.debug:
            if model_to_unload:
                fancy_print(self.console, f"Attempting to close TonicOllamaClient and unload specific model: {model_to_unload}...", style="dim blue")
            else:
                fancy_print(self.console, f"Attempting to close TonicOllamaClient and unload default models ({len(self.models_to_unload_on_close)})...", style="dim blue")

        ollama_client_instance = self.async_client

        if ollama_client_instance:
            models_to_attempt_unload = [model_to_unload] if model_to_unload else self.models_to_unload_on_close
            
            for model_name in models_to_attempt_unload:
                if not model_name: continue # Should not happen if logic is correct, but as a safeguard
                try:
                    if self.debug:
                        fancy_print(self.console, f"  Attempting to unload model: {model_name} (keep_alive='0s')", style="dim blue")
                    await ollama_client_instance.generate(
                        model=model_name,
                        prompt=".", 
                        options={"num_predict": 1}, 
                        keep_alive="0s"
                    )
                    if self.debug:
                        fancy_print(self.console, f"    Unload request sent for {model_name}.", style="dim green")
                except ResponseError as e:
                    if self.debug:
                        if e.status_code == 404:
                            fancy_print(self.console, f"    Model {model_name} not found or not loaded during close. (Error: {e})", style="dim yellow")
                        else:
                            fancy_print(self.console, f"    API error during unload attempt for model {model_name}: {e}", style="dim red")
                except Exception as e:
                    if self.debug:
                        fancy_print(self.console, f"    Unexpected error during unload attempt for model {model_name}: {e}", style="dim red")
            
            if hasattr(ollama_client_instance, '_client') and ollama_client_instance._client:
                try:
                    if self.debug:
                        fancy_print(self.console, "  Closing underlying HTTP client...", style="dim blue")
                    await ollama_client_instance._client.aclose()
                    if self.debug:
                        fancy_print(self.console, "    Underlying HTTP client closed.", style="dim green")
                except Exception as e:
                    if self.debug:
                        fancy_print(self.console, f"    Error closing underlying HTTP client: {e}", style="dim red")
            
            self.async_client = None # Clear the client instance

        if self.debug:
            fancy_print(self.console, "TonicOllamaClient close operation finished.", style="dim blue")


def create_client(
    base_url: str = "http://localhost:11434",
    max_server_startup_attempts: int = 3,
    debug: bool = False,
    console: Optional[Console] = None,
    concurrent_models: int = CONCURRENT_MODELS,
    models_to_unload_on_close: Optional[List[str]] = None,
    stream_responses: bool = False,
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
        concurrent_models=CONCURRENT_MODELS,
        models_to_unload_on_close=models_to_unload_on_close,
        stream_responses=stream_responses,
    )

def get_ollama_models_sync() -> List[str]:
    """Synchronously fetches locally available Ollama models."""
    try:
        import ollama
        models_info = ollama.list()
        if models_info and 'models' in models_info:
            return sorted([model['name'] for model in models_info['models'] if 'name' in model])
    except Exception:
        return []
    return []

__all__ = [
    "AsyncClient",
    "ChatResponse",
    "ClientConfig",
    "CONCURRENT_MODELS",
    "EmbeddingsResponse",
    "Message",
    "OllamaServerNotRunningError",
    "ResponseError",
    "TonicOllamaClient",
    "create_client",
    "get_ollama_models_sync",
    "fancy_print",
]

