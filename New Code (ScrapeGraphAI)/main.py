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

def main():
    st.title("Enhanced Event Scraper")
    st.write("Extract events from multiple URLs efficiently")

    # File upload for multiple URLs
    uploaded_file = st.file_uploader("Upload URLs file (CSV or TXT)", type=["csv", "txt"])
    
    # Manual URL input
    manual_url = st.text_input("Or enter a single URL:", 
                              placeholder="https://example.com/events")

    # Advanced settings in expander
    with st.expander("Advanced Settings"):
        batch_size = st.slider("Batch size", 1, 10, 5, 
                             help="Number of URLs to process simultaneously")
        timeout = st.slider("Timeout (seconds)", 10, 300, 60,
                          help="Maximum time to wait for each URL")

    if st.button("Start Scraping"):
        urls = []

        # Process uploaded file
        if uploaded_file:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                urls = df.iloc[:, 0].tolist()
            else:
                content = uploaded_file.read().decode()
                urls = [url.strip() for url in content.split('\n') if url.strip()]

        # Add manual URL if provided
        if manual_url:
            urls.append(manual_url)

        if urls:
            start_time = time.time()
            total_urls = len(urls)
            
            # Create progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback
            processed_urls = 0
            def update_progress():
                nonlocal processed_urls
                processed_urls += 1
                progress = processed_urls / total_urls
                progress_bar.progress(progress)
                status_text.text(f"Processed {processed_urls}/{total_urls} URLs...")

            try:
                with st.spinner('Scraping events...'):
                    # Process URLs
                    df = asyncio.run(process_urls(urls, update_progress))
                    
                    # Display results
                    if not df.empty:
                        end_time = time.time()
                        duration = round(end_time - start_time, 2)
                        
                        st.success(f"Successfully scraped {len(df)} events in {duration} seconds!")
                        
                        # Show results in tabs
                        tab1, tab2 = st.tabs(["Data Preview", "Statistics"])
                        
                        with tab1:
                            st.dataframe(df)
                            
                            # Download buttons
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "Download CSV",
                                csv,
                                f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                            
                        with tab2:
                            st.subheader("Scraping Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total URLs", total_urls)
                            with col2:
                                st.metric("Total Events", len(df))
                            with col3:
                                st.metric("Avg. Time per URL", f"{round(duration/total_urls, 2)}s")
                    else:
                        st.warning("No events were found in the provided URLs.")
                        
            except Exception as e:
                st.error(f"An error occurred during scraping: {str(e)}")
                logger.exception("Scraping error")
        else:
            st.warning("Please provide URLs either through file upload or manual input.")

if __name__ == "__main__":
    main()