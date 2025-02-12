import streamlit as st
import pandas as pd
import json
from langchain.agents import initialize_agent, AgentType
from langchain_scrapegraph.tools import SmartScraperTool
from langchain_openai import ChatOpenAI
from datetime import datetime
import os

# Set your API keys
os.environ["SGAI_API_KEY"] = "PASTE YOUR HERE Search Grpah API Key"
os.environ['OPENAI_API_KEY'] = "Paste here your OPENAI APIKEY"

def init_scraper():
    tools = [SmartScraperTool()]
    return initialize_agent(
        tools=tools,
        llm=ChatOpenAI(temperature=0),
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

def extract_events(url):
    agent = init_scraper()
    
    prompt = f"""
    Visit {url}. Extract events and return them in the following EXACT JSON format, with ALL fields required for each event:

    {{
        "events": [
            {{
                "event_name": "Event Name",
                "event_venue_name": "Venue Name",
                "event_venue_address": "Full Venue Address",
                "event_start_date": "DD/MM/YYYY",
                "event_start_time": "HH:MM",
                "event_end_date": "DD/MM/YYYY",
                "event_end_time": "HH:MM",
                "event_category": "Event Category",
                "event_description": "Event Description",
                "event_image_url": "Full Image URL",
                "event_link": "Full Event URL"
            }}
        ]
    }}

    Make sure to:
    1. Include ALL fields for EACH event
    2. Use EXACTLY these field names
    3. Format dates as DD/MM/YYYY
    4. Format times as HH:MM
    5. Include full URLs for images and links
    6. Do not combine date and time fields
    7. Extract location details separately into venue name and address
    """
    
    response = agent.run(prompt)
    return response

def process_response(response):
    try:
        # Try to parse the JSON response
        data = json.loads(response)
        
        # If the response is already in the correct format
        if isinstance(data, dict) and "events" in data:
            events = data["events"]
        # If the response is a list of events
        elif isinstance(data, list):
            events = data
        else:
            raise ValueError("Unexpected response format")

        # Ensure all required fields are present
        required_fields = [
            'event_name', 'event_venue_name', 'event_venue_address',
            'event_start_date', 'event_start_time', 'event_end_date',
            'event_end_time', 'event_category', 'event_description',
            'event_image_url', 'event_link'
        ]

        # Process each event to ensure all fields exist
        processed_events = []
        for event in events:
            processed_event = {field: event.get(field, '') for field in required_fields}
            processed_events.append(processed_event)

        return pd.DataFrame(processed_events)
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        raise

def main():
    st.title("Event Scraper")
    
    url = st.text_input("Enter URL to scrape:", 
                       placeholder="https://example.com/events")
    
    if st.button("Start Scraping"):
        if url:
            try:
                with st.spinner("Scraping events..."):
                    # Extract events
                    response = extract_events(url)
                    
                    # Process response into DataFrame
                    df = process_response(response)
                    
                    # Show results
                    st.success(f"Successfully scraped {len(df)} events!")
                    st.dataframe(df)
                    
                    # Download options
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a URL")

if __name__ == "__main__":
    main()