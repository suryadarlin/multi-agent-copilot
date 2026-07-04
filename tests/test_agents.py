"""
Test file for agents
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.planner_agent import PlannerAgent


def test_planner_exists():
    """Test planner agent exists"""
    planner = PlannerAgent()
    assert planner is not None
    print(f"✅ Planner created: {planner}")


def test_planner_name():
    """Test planner name"""
    planner = PlannerAgent()
    assert planner.name == "Planner"
    print("✅ Planner name correct")


def test_planner_skills():
    """Test planner has skills"""
    planner = PlannerAgent()
    skills = planner.get_skills()
    assert len(skills) > 0
    print(f"✅ Planner skills: {skills}")


@pytest.mark.asyncio
async def test_planner_execute():
    """Test planner execute"""
    planner = PlannerAgent()
    
    task = {"request": "Build a REST API"}
    response = await planner.execute(task)
    
    assert response.success == True
    print(f"✅ Planner execute successful")