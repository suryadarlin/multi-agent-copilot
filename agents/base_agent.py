"""
Base agent class for all agents to inherit from
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel
import json
from datetime import datetime


class AgentResponse(BaseModel):
    """Response from an agent"""
    success: bool
    data: Any
    error: Optional[str] = None
    agent_name: str
    timestamp: str = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, llm=None):
        """
        Initialize base agent
        
        Args:
            name: Name of the agent
            llm: Language model instance (optional)
        """
        self.name = name
        self.llm = llm
        self.skills = []
        self.created_at = datetime.now().isoformat()
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> AgentResponse:
        """
        Execute a task
        
        Args:
            task: Task dictionary with instructions
            
        Returns:
            AgentResponse with results or error
        """
        pass
    
    def add_skill(self, skill_name: str):
        """
        Add a skill to this agent
        
        Args:
            skill_name: Name of the skill
        """
        if skill_name not in self.skills:
            self.skills.append(skill_name)
    
    def get_skills(self) -> list:
        """Get list of agent skills"""
        return self.skills
    
    def __repr__(self):
        return f"{self.name} Agent (Skills: {', '.join(self.skills)})"
    
    def __str__(self):
        return self.__repr__()