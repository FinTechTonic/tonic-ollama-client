import pytest
import pytest_asyncio
import uuid
from tonic_ollama_client import TonicOllamaClient

@pytest_asyncio.fixture
async def client():
    client = TonicOllamaClient(debug=True)
    yield client

@pytest.mark.asyncio
async def test_create_conversation_auto_id(client):
    """Test creating conversation with auto-generated ID."""
    conv_id = await client.create_conversation()
    
    assert isinstance(conv_id, str)
    assert len(conv_id) > 0
    
    uuid.UUID(conv_id)  # Will raise if not valid UUID
    
    assert conv_id in client.list_conversations()
    assert client.get_conversation(conv_id) == []

@pytest.mark.asyncio
async def test_create_conversation_custom_id(client):
    """Test creating conversation with custom ID."""
    custom_id = "my-custom-conversation-id"
    conv_id = await client.create_conversation(custom_id)
    
    assert conv_id == custom_id
    assert custom_id in client.list_conversations()
    assert client.get_conversation(custom_id) == []

@pytest.mark.asyncio
async def test_create_duplicate_conversation(client):
    """Test creating conversation with duplicate ID."""
    conv_id = "duplicate-test"
    
    result1 = await client.create_conversation(conv_id)
    assert result1 == conv_id
    
    result2 = await client.create_conversation(conv_id)
    assert result2 == conv_id
    
    conversations = client.list_conversations()
    assert conversations.count(conv_id) == 1

@pytest.mark.asyncio
async def test_conversation_operations_comprehensive(client):
    """Test all conversation operations comprehensively."""
    assert client.list_conversations() == []
    
    conv1 = await client.create_conversation("conv1")
    conv2 = await client.create_conversation("conv2")
    conv3 = await client.create_conversation()  # Auto ID
    
    conversations = client.list_conversations()
    assert len(conversations) == 3
    assert "conv1" in conversations
    assert "conv2" in conversations
    assert conv3 in conversations
    
    client.conversations[conv1].append({"role": "user", "content": "Hello"})
    client.conversations[conv1].append({"role": "assistant", "content": "Hi there"})
    client.conversations[conv2].append({"role": "user", "content": "Test"})
    
    conv1_messages = client.get_conversation(conv1)
    assert len(conv1_messages) == 2
    assert conv1_messages[0]["content"] == "Hello"
    assert conv1_messages[1]["content"] == "Hi there"
    
    conv2_messages = client.get_conversation(conv2)
    assert len(conv2_messages) == 1
    assert conv2_messages[0]["content"] == "Test"
    
    conv3_messages = client.get_conversation(conv3)
    assert len(conv3_messages) == 0
    
    client.clear_conversation(conv1)
    assert client.get_conversation(conv1) == []
    assert len(client.get_conversation(conv2)) == 1  # Other conversations unaffected
    
    client.delete_conversation(conv2)
    remaining_conversations = client.list_conversations()
    assert "conv2" not in remaining_conversations
    assert "conv1" in remaining_conversations
    assert conv3 in remaining_conversations
    
    with pytest.raises(ValueError, match="does not exist"):
        client.get_conversation(conv2)

@pytest.mark.asyncio
async def test_conversation_error_handling(client):
    """Test error handling for conversation operations."""
    with pytest.raises(ValueError, match="does not exist"):
        client.get_conversation("nonexistent")
    
    with pytest.raises(ValueError, match="does not exist"):
        client.clear_conversation("nonexistent")
    
    with pytest.raises(ValueError, match="does not exist"):
        client.delete_conversation("nonexistent")

@pytest.mark.asyncio
async def test_conversation_message_types(client):
    """Test conversation with different message types."""
    conv_id = await client.create_conversation("message-types-test")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "I don't have access to weather data."}
    ]
    
    for message in messages:
        client.conversations[conv_id].append(message)
    
    stored_messages = client.get_conversation(conv_id)
    assert len(stored_messages) == 5
    
    assert stored_messages[0]["role"] == "system"
    assert stored_messages[1]["role"] == "user"
    assert stored_messages[2]["role"] == "assistant"
    assert stored_messages[3]["role"] == "user"
    assert stored_messages[4]["role"] == "assistant"
    
    assert "helpful assistant" in stored_messages[0]["content"]
    assert "weather" in stored_messages[3]["content"]

@pytest.mark.asyncio
async def test_conversation_isolation(client):
    """Test that conversations are properly isolated."""
    conv1 = await client.create_conversation("isolation-test-1")
    conv2 = await client.create_conversation("isolation-test-2")
    
    client.conversations[conv1].append({"role": "user", "content": "Message for conv1"})
    client.conversations[conv2].append({"role": "user", "content": "Message for conv2"})
    
    conv1_messages = client.get_conversation(conv1)
    conv2_messages = client.get_conversation(conv2)
    
    assert len(conv1_messages) == 1
    assert len(conv2_messages) == 1
    assert conv1_messages[0]["content"] != conv2_messages[0]["content"]
    
    client.clear_conversation(conv1)
    
    assert client.get_conversation(conv1) == []
    assert len(client.get_conversation(conv2)) == 1

@pytest.mark.asyncio
async def test_large_conversation(client):
    """Test handling of large conversations."""
    conv_id = await client.create_conversation("large-conversation")
    
    num_messages = 100
    for i in range(num_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i} from {role}"
        client.conversations[conv_id].append({"role": role, "content": content})
    
    messages = client.get_conversation(conv_id)
    assert len(messages) == num_messages
    
    assert messages[0]["content"] == "Message 0 from user"
    assert messages[-1]["content"] == f"Message {num_messages-1} from assistant"
    
    for i, message in enumerate(messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == expected_role

@pytest.mark.asyncio 
async def test_conversation_with_special_characters(client):
    """Test conversations with special characters and unicode."""
    conv_id = await client.create_conversation("special-chars-test")
    
    special_messages = [
        {"role": "user", "content": "Hello! 🤖 How are you?"},
        {"role": "assistant", "content": "I'm doing well! 😊 Thanks for asking."},
        {"role": "user", "content": "Can you handle émojis and ñoñó?"},
        {"role": "assistant", "content": "Yes! I can handle unicode: 中文, العربية, 🌟"},
        {"role": "user", "content": 'Quote test: "Hello" and \'world\''},
        {"role": "assistant", "content": "Newline\ntest\nwith\nmultiple\nlines"}
    ]
    
    for message in special_messages:
        client.conversations[conv_id].append(message)
    
    stored_messages = client.get_conversation(conv_id)
    assert len(stored_messages) == 6
    
    assert "🤖" in stored_messages[0]["content"]
    assert "😊" in stored_messages[1]["content"]
    assert "ñoñó" in stored_messages[2]["content"]
    assert "中文" in stored_messages[3]["content"]
    assert '"Hello"' in stored_messages[4]["content"]
    assert "\n" in stored_messages[5]["content"]
