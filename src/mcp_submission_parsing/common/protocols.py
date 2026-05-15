"""Communication protocols between MCP components"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
from dataclasses import dataclass, asdict


@dataclass
class User:
    name: str
    age: int
    email: str = ""  # Optional field with default value

user = User(name="Alice", age=30)
print(user)  # Output: User(name='Alice', age=30, email='')   

class MessageType(str, Enum):
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    ERROR = "ERROR"
    STREAM = "STREAM"
    HEARTBEAT = "HEARTBEAT"


class AgentEndpoint(str, Enum):
    """Available agent endpoints"""
    PARSE_CLAIM = "agent/parse"
    LOOKUP_POLICY = "agent/policy"
    CHECK_RISK = "agent/risk"
    BUILD_FEATURES = "agent/features"
    DETECT_FRAUD = "agent/fraud"
    COMPLETE_PIPELINE = "agent/complete"
    HEALTH_CHECK = "agent/health"


@dataclass
class Message:
    """Standard message format for MCP communication"""
    message_id: str
    type: MessageType
    endpoint: AgentEndpoint
    payload: Dict[str, Any]
    timestamp: str
    sender: str
    correlation_id: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        data = json.loads(json_str)
        return cls(**data)


class MCPClientProtocol:
    """Base protocol for MCP clients"""
    
    async def connect(self) -> bool:
        """Establish connection to server"""
        pass
    
    async def send_request(self, endpoint: AgentEndpoint, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send request and wait for response"""
        pass
    
    async def stream_request(self, endpoint: AgentEndpoint, payload: Dict[str, Any]):
        """Send request and receive streaming response"""
        pass
    
    async def disconnect(self):
        """Close connection"""
        pass


class MCPServerProtocol:
    """Base protocol for MCP servers"""
    
    async def start(self):
        """Start the server"""
        pass
    
    async def stop(self):
        """Stop the server"""
        pass
    
    async def handle_request(self, message: Message) -> Message:
        """Handle incoming request"""
        pass
    
    async def register_agent(self, agent_name: str, agent_handler):
        """Register an agent with the server"""
        pass