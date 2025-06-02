import pytest
import pytest_asyncio
from unittest.mock import patch
from rich.console import Console
from tonic_ollama_client import TonicOllamaClient, create_client, ClientConfig

class TestClientConfiguration:
        
    def test_default_initialization(self):
        client = TonicOllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.max_readiness_attempts == 3
        assert client.debug is False
        assert isinstance(client.console, Console)
        assert client.conversations == {}

    def test_custom_initialization(self):
        custom_console = Console()
        client = TonicOllamaClient(
            base_url="http://custom-url:12345",
            max_readiness_attempts=5,
            debug=True,
            console=custom_console,
        )
        assert client.base_url == "http://custom-url:12345"
        assert client.max_readiness_attempts == 5
        assert client.debug is True
        assert client.console == custom_console

    def test_client_config_model_defaults(self):
        config = ClientConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.max_readiness_attempts == 3
        assert config.debug is False

    def test_client_config_model_custom(self):
        config = ClientConfig(
            base_url="http://another-url:54321",
            max_readiness_attempts=10,
            debug=True
        )
        assert config.base_url == "http://another-url:54321"
        assert config.max_readiness_attempts == 10
        assert config.debug is True

    def test_create_client_defaults(self):
        client = create_client()
        assert client.base_url == "http://localhost:11434"
        assert client.max_readiness_attempts == 3
        assert client.debug is False
        assert isinstance(client.console, Console)

    def test_create_client_custom_params(self):
        custom_console = Console()
        client = create_client(
            base_url="http://custom-ollama:11434",
            max_readiness_attempts=5,
            debug=True,
            console=custom_console
        )
        assert client.base_url == "http://custom-ollama:11434"
        assert client.max_readiness_attempts == 5
        assert client.debug is True
        assert client.console == custom_console

    def test_client_config_passed_to_client(self):
        """Test TonicOllamaClient uses parameters as if from a config."""
        config = ClientConfig(
            base_url="http://config-test:1122", 
            max_readiness_attempts=7,
            debug=True
        )
        
        client_from_config_values = TonicOllamaClient(
            base_url=config.base_url,
            max_readiness_attempts=config.max_readiness_attempts,
            debug=config.debug
        )

        assert client_from_config_values.base_url == config.base_url
        assert client_from_config_values.max_readiness_attempts == config.max_readiness_attempts
        assert client_from_config_values.debug == config.debug

    @patch('tonic_ollama_client.Console')
    def test_create_client_console_handling(self, MockConsole):
        """Test create_client console handling."""
        MockConsole.reset_mock()
        client1 = create_client()
        MockConsole.assert_called_once() 
        assert isinstance(client1.console, MockConsole)

        MockConsole.reset_mock()
        my_console = MockConsole()
        client2 = create_client(console=my_console)
        assert client2.console == my_console
