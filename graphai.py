import streamlit as st
import pandas as pd
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import io
from dotenv import load_dotenv
import os
from scrapegraphai.graphs import SmartScraperGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import nest_asyncio

# Enable nested asyncio for Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv("var.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Event:
    event_name: str
    venue_name: str = ""
    venue_address: str = ""
    start_date: str = ""
    start_time: str = ""
    end_date: str = ""
    end_time: str = ""
    category: str = ""
    event_link: str = ""
    description: str = ""
    image_url: str = ""
    source_url: str = ""
    price: str = ""
    organizer: str = ""

class EventExtractor:
    def __init__(self, api_key: str, model_type: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_type = model_type
        self.events_cache: Dict[str, List[Event]] = {}

    def create_extraction_prompt(self) -> str:
        """Create detailed prompt for event extraction"""
        return """Extract all events from this webpage. For each event, provide:
        {
            "event_name": "string (required)",
            "venue_name": "string",
            "venue_address": "string",
            "start_date": "YYYY-MM-DD format",
            "start_time": "HH:MM format",
            "end_date": "YYYY-MM-DD format",
            "end_time": "HH:MM format",
            "category": "string",
            "description": "string",
            "price": "string with currency",
            "organizer": "string",
            "image_url": "URL if available",
            "event_link": "URL to event details"
        }
        Return the data as a JSON array of event objects.
        Also analyze any images found for additional event information."""

    async def process_single_url(self, url: str) -> List[Event]:
        """Process a single URL using SmartScraperGraph"""
        try:
            # Configure the graph
            graph_config = {
                "llm": {
                    "api_key": self.api_key,
                    "model": f"openai/{self.model_type}",
                    "temperature": 0,
                },
                "verbose": True,
                "headless": True,
                "playwright": {
                    "timeout": 30000,  # 30 seconds timeout
                    "wait_until": "networkidle"
                }
            }

            # Create and run the scraper
            scraper = SmartScraperGraph(
                prompt=self.create_extraction_prompt(),
                source=url,
                config=graph_config
            )

            # Run the scraper
            result = await scraper.run()  # Use arun() instead of run()

            # Parse the results
            events = []
            if isinstance(result, dict) and "answer" in result:
                try:
                    events_data = json.loads(result["answer"])
                    if isinstance(events_data, list):
                        for event_data in events_data:
                            if event_data.get('event_name'):
                                event = Event(
                                    event_name=event_data.get('event_name', ''),
                                    venue_name=event_data.get('venue_name', ''),
                                    venue_address=event_data.get('venue_address', ''),
                                    start_date=event_data.get('start_date', ''),
                                    start_time=event_data.get('start_time', ''),
                                    end_date=event_data.get('end_date', ''),
                                    end_time=event_data.get('end_time', ''),
                                    category=event_data.get('category', ''),
                                    description=event_data.get('description', ''),
                                    price=event_data.get('price', ''),
                                    organizer=event_data.get('organizer', ''),
                                    image_url=event_data.get('image_url', ''),
                                    event_link=event_data.get('event_link', '') or url,
                                    source_url=url
                                )
                                events.append(event)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing events data for {url}")

            self.events_cache[url] = events
            return events

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return []

    async def process_urls(self, urls: List[str]) -> pd.DataFrame:
        """Process multiple URLs and return results as DataFrame"""
        try:
            # Process all URLs concurrently
            tasks = [self.process_single_url(url) for url in urls]
            all_results = await asyncio.gather(*tasks)
            
            # Flatten results
            flattened_events = [
                event for url_results in all_results 
                for event in url_results
            ]
            
            # Convert to DataFrame
            df = pd.DataFrame([vars(event) for event in flattened_events])
            if not df.empty:
                # Clean and sort data
                df = df.drop_duplicates(
                    subset=['event_name', 'start_date', 'venue_name'], 
                    keep='first'
                )
                df = df.fillna('')
                
                # Convert dates for sorting
                for col in ['start_date', 'end_date']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                
                df = df.sort_values('start_date')
                
                # Convert back to string format
                for col in ['start_date', 'end_date']:
                    if col in df.columns:
                        df[col] = df[col].dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing URLs: {str(e)}")
            return pd.DataFrame()

def create_async_loop():
    """Create a new event loop"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

async def async_scrape(urls: List[str], api_key: str, model_type: str) -> pd.DataFrame:
    """Async wrapper for scraping operation"""
    extractor = EventExtractor(api_key=api_key, model_type=model_type)
    return await extractor.process_urls(urls)


def main():
    st.set_page_config(page_title="Event Scraper Pro", layout="wide")
    st.title("🎯 Event Scraper Pro")

    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set it in your environment variables.")
        return

    with st.sidebar:
        st.header("⚙️ Settings")
        model_type = st.selectbox(
            "Select Model",
            ["gpt-4o-mini", "gpt-3.5-turbo"],
            help="Select the OpenAI model to use"
        )
        
        st.markdown("---")
        st.header("📊 Visualization")
        chart_type = st.selectbox(
            "Select chart type",
            ["Bar", "Line", "Calendar Heat Map"],
            key="chart_type"
        )

    # File upload and URL input section
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "📂 Upload URLs file (CSV or TXT)",
            type=["csv", "txt"],
            help="Upload a file containing URLs, one per line"
        )
        
    with col2:
        manual_url = st.text_input(
            "🔗 Or enter a single URL:",
            placeholder="https://example.com/events"
        )

    if st.button("🚀 Start Scraping", type="primary"):
        urls = []
        
        if uploaded_file:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                urls = df.iloc[:, 0].tolist()
            else:
                content = uploaded_file.read().decode()
                urls = [url.strip() for url in content.split('\n') if url.strip()]

        if manual_url:
            urls.append(manual_url)

        if urls:
            try:
                with st.spinner("🔄 Scraping events..."):
                    start_time = time.time()
                    
                    # Initialize progress
                    progress_bar = st.progress(0, text="Starting scraping...")
                    
                    # Create async loop and run scraping
                    loop = create_async_loop()
                    df = loop.run_until_complete(async_scrape(urls, OPENAI_API_KEY, model_type))
                    
                    if not df.empty:
                        end_time = time.time()
                        duration = round(end_time - start_time, 2)
                        
                        # Update progress
                        progress_bar.progress(100, text="Scraping completed!")
                        
                        # Success message and stats
                        st.success(f"✅ Successfully scraped {len(df)} events in {duration} seconds!")
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🎯 Total Events", len(df))
                        with col2:
                            st.metric("🏟️ Unique Venues", df['venue_name'].nunique())
                        with col3:
                            st.metric("📸 Events with Images", df['image_url'].notna().sum())
                        with col4:
                            st.metric("🌐 URLs Processed", len(urls))

                        # Data Quality Analysis
                        with st.expander("📊 Data Quality Analysis"):
                            completion_rates = pd.DataFrame({
                                'Field': df.columns,
                                'Completion Rate': [
                                    f"{(df[col].notna().sum() / len(df) * 100):.1f}%" 
                                    for col in df.columns
                                ]
                            })
                            st.write("Field Completion Rates:")
                            st.dataframe(
                                completion_rates,
                                hide_index=True,
                                use_container_width=True
                            )

                        # Filters
                        st.subheader("🔍 Filter Events")
                        col1, col2, col3 = st.columns(3)
                        
                        filtered_df = df.copy()
                        
                        with col1:
                            if 'start_date' in df.columns and not df['start_date'].empty:
                                date_range = st.date_input(
                                    "Filter by Date Range",
                                    value=(
                                        pd.to_datetime(df['start_date'].min()).date(),
                                        pd.to_datetime(df['start_date'].max()).date()
                                    ),
                                    key="date_filter"
                                )
                                if len(date_range) == 2:
                                    start_date, end_date = date_range
                                    mask = (pd.to_datetime(filtered_df['start_date']).dt.date >= start_date) & \
                                          (pd.to_datetime(filtered_df['start_date']).dt.date <= end_date)
                                    filtered_df = filtered_df[mask]
                        
                        with col2:
                            if 'category' in df.columns and not df['category'].empty:
                                categories = sorted(df['category'].unique())
                                selected_categories = st.multiselect(
                                    "Filter by Category",
                                    options=categories,
                                    key="category_filter"
                                )
                                if selected_categories:
                                    filtered_df = filtered_df[
                                        filtered_df['category'].isin(selected_categories)
                                    ]
                        
                        with col3:
                            if 'venue_name' in df.columns and not df['venue_name'].empty:
                                venues = sorted(df['venue_name'].unique())
                                selected_venues = st.multiselect(
                                    "Filter by Venue",
                                    options=venues,
                                    key="venue_filter"
                                )
                                if selected_venues:
                                    filtered_df = filtered_df[
                                        filtered_df['venue_name'].isin(selected_venues)
                                    ]

                        # Display filtered results
                        st.subheader("📋 Events Data")
                        st.dataframe(
                            filtered_df,
                            column_config={
                                "event_link": st.column_config.LinkColumn("Event Link"),
                                "image_url": st.column_config.ImageColumn("Image Preview"),
                                "description": st.column_config.TextColumn(
                                    "Description",
                                    max_chars=50,
                                    help="Click to expand"
                                ),
                            },
                            hide_index=True,
                            use_container_width=True
                        )

                        # Export options
                        st.subheader("💾 Export Data")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            csv = filtered_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )
                        
                        with col2:
                            # Excel export with formatting
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                filtered_df.to_excel(writer, sheet_name='Events', index=False)
                                workbook = writer.book
                                worksheet = writer.sheets['Events']
                                
                                # Format headers
                                header_format = workbook.add_format({
                                    'bold': True,
                                    'text_wrap': True,
                                    'valign': 'top',
                                    'fg_color': '#D9EAD3',
                                    'border': 1
                                })
                                
                                for col_num, value in enumerate(filtered_df.columns.values):
                                    worksheet.write(0, col_num, value, header_format)
                                    width = max(
                                        len(str(value)) + 2,
                                        filtered_df[value].astype(str).str.len().max() + 2
                                    )
                                    worksheet.set_column(col_num, col_num, min(width, 50))
                            
                            excel_data = output.getvalue()
                            st.download_button(
                                label="📥 Download Excel",
                                data=excel_data,
                                file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                        with col3:
                            # JSON export
                            json_str = filtered_df.to_json(orient='records', indent=2)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_str,
                                file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

                        # Add visualization based on chart type
                        if len(filtered_df) > 0:
                            st.subheader("📈 Event Analysis")
                            if chart_type == "Bar":
                                if 'category' in filtered_df.columns:
                                    st.bar_chart(filtered_df['category'].value_counts())
                            elif chart_type == "Line":
                                if 'start_date' in filtered_df.columns:
                                    events_by_date = filtered_df['start_date'].value_counts().sort_index()
                                    st.line_chart(events_by_date)
                            elif chart_type == "Calendar Heat Map":
                                if 'start_date' in filtered_df.columns:
                                    try:
                                        import calplot
                                        events_by_date = filtered_df['start_date'].value_counts()
                                        fig = calplot.calplot(events_by_date)
                                        st.pyplot(fig)
                                    except ImportError:
                                        st.warning("Please install calplot for calendar visualization: pip install calplot")
                    else:
                        st.error("❌ No events found. Please check the URLs and try again.")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                logger.exception("Scraping error")
        else:
            st.warning("⚠️ Please provide at least one URL to scrape.")

if __name__ == "__main__":
    main()