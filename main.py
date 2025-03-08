import streamlit as st
import pandas as pd
import json
from langchain.agents import initialize_agent, AgentType
from langchain_scrapegraph.tools import SmartScraperTool
from langchain_openai import ChatOpenAI
from datetime import datetime
import asyncio
import time
from typing import List, Dict
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Set API keys
os.environ["SGAI_API_KEY"] = "PASTE YOUR SCRAPE GRAPH API KEY HERE"
os.environ['OPENAI_API_KEY'] = "PASTE OPENAI KEY HERE"

def init_scraper():
    tools = [SmartScraperTool()]
    return initialize_agent(
        tools=tools,
        llm=ChatOpenAI(temperature=0),
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

def extract_events(url: str) -> str:
    agent = init_scraper()
    
    try:
        action_input = {
            "website_url": url,
            "user_prompt": "Extract all events with the following information: event name, venue name, address, start date (DD/MM/YYYY), start time (HH:MM), end date (DD/MM/YYYY), end time (HH:MM), category, description, image URL, and event URL."
        }
        
        # Use invoke with the correct action format
        response = agent.invoke({
            "input": {
                "action": "SmartScraper",
                "action_input": action_input
            }
        })
        
        # Extract and format the events data
        if isinstance(response, dict) and "output" in response:
            events_data = {
                "events": []
            }
            
            # Parse and structure the extracted data
            try:
                extracted_data = response["output"]
                if isinstance(extracted_data, str):
                    # Try to parse if it's a JSON string
                    try:
                        extracted_data = json.loads(extracted_data)
                    except:
                        pass
                
                # Ensure we have a list of events
                if isinstance(extracted_data, dict) and "events" in extracted_data:
                    events_data = extracted_data
                elif isinstance(extracted_data, list):
                    events_data["events"] = extracted_data
                
                return json.dumps(events_data)
            except Exception as e:
                logger.error(f"Error parsing extracted data: {str(e)}")
                raise
                
        return json.dumps({"events": []})
    except Exception as e:
        logger.error(f"Error extracting events from {url}: {str(e)}")
        raise

def process_response(response: str) -> pd.DataFrame:
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
            # Convert the event to the required format
            processed_event = {
                'event_name': event.get('name', event.get('event_name', '')),
                'event_venue_name': event.get('venue', event.get('event_venue_name', '')),
                'event_venue_address': event.get('address', event.get('event_venue_address', '')),
                'event_start_date': event.get('start_date', event.get('event_start_date', '')),
                'event_start_time': event.get('start_time', event.get('event_start_time', '')),
                'event_end_date': event.get('end_date', event.get('event_end_date', '')),
                'event_end_time': event.get('end_time', event.get('event_end_time', '')),
                'event_category': event.get('category', event.get('event_category', '')),
                'event_description': event.get('description', event.get('event_description', '')),
                'event_image_url': event.get('image_url', event.get('event_image_url', '')),
                'event_link': event.get('url', event.get('event_link', ''))
            }
            processed_events.append(processed_event)

        df = pd.DataFrame(processed_events)
        return df.drop_duplicates(subset=['event_name', 'event_start_date'], keep='first')
    except Exception as e:
        logger.error(f"Error processing response: {str(e)}")
        raise

async def process_url(url: str, progress_callback=None) -> pd.DataFrame:
    try:
        response = extract_events(url)
        df = process_response(response)
        if progress_callback:
            progress_callback()
        return df
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return pd.DataFrame()

async def process_urls(urls: List[str], progress_callback=None) -> pd.DataFrame:
    tasks = [process_url(url, progress_callback) for url in urls]
    results = await asyncio.gather(*tasks)
    
    # Combine all DataFrames
    combined_df = pd.concat(results, ignore_index=True)
    return combined_df.drop_duplicates(subset=['event_name', 'event_start_date'], keep='first')