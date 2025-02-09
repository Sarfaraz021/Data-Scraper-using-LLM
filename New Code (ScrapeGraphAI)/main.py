from langchain.agents import initialize_agent, AgentType
from langchain_scrapegraph.tools import SmartScraperTool
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

# Initialize tools
tools = [
    SmartScraperTool(),
]

# Create an agent
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
response = agent.run("""
    Visit https://www.visitmonaco.com/fr/22107/tout-ce-qu-il-se-passe-a-monaco, List all events from this webpage. For each event provide the following details:
            event_name
            venue_name
            venue_address
            start_date
            start_time
            end_date
            end_time
            category
            description
            price
            organizer
            image_url
            event_link

Make sure to must scrape image/ event banner url and event link.

Return the data as a structured JSON array with each event as an object containing all available fields."""
)

print(response)