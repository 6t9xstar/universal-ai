from __future__ import annotations

import asyncio

from universal_ai import AI, Agent, Team, tool


@tool
def search_database(query: str) -> str:
    """Search the internal database for information."""
    return f"Results for: {query}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to}: {subject}"


async def multi_agent_example():
    ai = AI()

    researcher = Agent(name="researcher", system_prompt="You are a research expert.")
    writer = Agent(name="writer", system_prompt="You are a skilled writer.")
    reviewer = Agent(name="reviewer", system_prompt="You are a thorough reviewer.")

    team = Team(researcher, writer, reviewer)

    response = await team.run(ai, "Write an article about AI ethics.")
    print(f"Final output:\n{response}")


if __name__ == "__main__":
    asyncio.run(multi_agent_example())
