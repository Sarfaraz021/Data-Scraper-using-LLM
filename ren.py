import streamlit as st
import pandas as pd
from langchain_anthropic import ChatAnthropic
from bs4 import BeautifulSoup, Tag
import aiohttp
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
import re
import io
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from collections import defaultdict

load_dotenv("var.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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

class EventExtractor:
    def __init__(self):
        # self.llm = ChatAnthropic(model='claude-3-sonnet-20240229')
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}',
        ]
        self.time_pattern = r'\d{1,2}:\d{2}'
    
    def _extract_dates(self, text: str) -> Tuple[str, str]:
        """Extract start and end dates from text"""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        if not dates:
            return "", ""
        
        return dates[0], dates[-1] if len(dates) > 1 else ""

    def _extract_times(self, text: str) -> Tuple[str, str]:
        """Extract start and end times from text"""
        times = re.findall(self.time_pattern, text)
        if not times:
            return "", ""
        return times[0], times[-1] if len(times) > 1 else ""

    async def extract_event_data(self, content: str, url: str) -> List[Event]:
        try:
            system_message = {
                "role": "system",
                "content": "Extract structured event data from HTML. Return results as a JSON array."
            }
            user_message = {
                "role": "user", 
                "content": f"""Extract all events from this HTML. For each event, provide:
                    - event_name (required)
                    - venue_name
                    - venue_address 
                    - start_date (YYYY-MM-DD)
                    - start_time (HH:MM)
                    - end_date (YYYY-MM-DD)
                    - end_time (HH:MM)
                    - category
                    - description
                    
                    Content: {content[:15000]}"""
            }
            
            response = await self.llm.ainvoke([system_message, user_message])
            return self._parse_llm_response(response.content, url)
            
        except Exception as e:
            logger.error(f"Error extracting events from {url}: {str(e)}")
            return []
        
    def _parse_llm_response(self, response: str, url: str) -> List[Event]:
        """Parse LLM response into Event objects"""
        try:
            if not isinstance(response, str):
                return []

            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response.replace('\n', ''))
            if not json_match:
                return []
            
            json_str = json_match.group()
            events_data = json.loads(json_str)
            
            events = []
            for data in events_data:
                if not data.get('event_name'):
                    continue
                    
                event = Event(
                    event_name=data.get('event_name', ''),
                    venue_name=data.get('venue_name', ''),
                    venue_address=data.get('venue_address', ''),
                    start_date=data.get('start_date', ''),
                    start_time=data.get('start_time', ''),
                    end_date=data.get('end_date', ''),
                    end_time=data.get('end_time', ''),
                    category=data.get('category', ''),
                    description=data.get('description', ''),
                    source_url=url
                )
                events.append(event)
                
            return events
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return []

class ContentParser:
    """Handles HTML parsing and content extraction"""
    
    def __init__(self):
        self.event_keywords = {
            'event', 'agenda', 'calendar', 'programme', 'schedule',
            'concert', 'exhibition', 'show', 'performance', 'festival',
            'workshop', 'conference', 'meeting', 'seminar', 'exposition'
        }

    def extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract relevant image URLs"""
        images = []
        exclude_patterns = re.compile(r'icon|logo|banner|background|social|share', re.I)
        
        for img in soup.find_all('img'):
            for attr in ['src', 'data-src', 'data-lazy-src']:
                src = img.get(attr)
                if not src:
                    continue
                    
                if exclude_patterns.search(src):
                    continue
                    
                # Check image dimensions if available
                width = img.get('width', '0')
                height = img.get('height', '0')
                try:
                    if int(width) < 100 or int(height) < 100:
                        continue
                except ValueError:
                    pass
                
                absolute_url = urljoin(base_url, src)
                images.append(absolute_url)
                break
                
        return images

    def extract_event_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract unique event links per event"""
        event_links = []
        base_domain = urlparse(base_url).netloc
        
        # Find all event containers
        event_containers = soup.find_all(class_=re.compile(r'event|agenda-item|calendar-item', re.I))
        if not event_containers:
            event_containers = [soup]
        
        for container in event_containers:
            event_link = None
            
            # Try to find specific event link within each container
            for link in container.find_all('a', href=True):
                href = link['href']
                if not href or href.startswith(('javascript:', '#', 'mailto:')):
                    continue
                    
                text = link.get_text(strip=True).lower()
                if any(keyword in text or keyword in href.lower() for keyword in self.event_keywords):
                    absolute_url = urljoin(base_url, href)
                    if urlparse(absolute_url).netloc == base_domain:
                        event_link = absolute_url
                        break
            
            if event_link and event_link not in event_links:
                event_links.append(event_link)
                
        return event_links

    def extract_structured_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract event data from schema.org markup"""
        events = []
        
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    if data.get('@type') in ['Event', 'EventSeries']:
                        event = self._parse_schema_event(data)
                        if event:
                            events.append(event)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') in ['Event', 'EventSeries']:
                            event = self._parse_schema_event(item)
                            if event:
                                events.append(event)
            except Exception as e:
                logger.error(f"Error parsing structured data: {str(e)}")
                continue
                
        return events

    def _parse_schema_event(self, data: Dict) -> Optional[Dict]:
        """Parse schema.org Event data"""
        try:
            event = {
                'event_name': data.get('name', ''),
                'start_date': '',
                'end_date': '',
                'description': data.get('description', '')
            }
            
            # Handle dates
            start_date = data.get('startDate')
            if start_date:
                event['start_date'] = start_date.split('T')[0]
                
            end_date = data.get('endDate')
            if end_date:
                event['end_date'] = end_date.split('T')[0]
            
            # Handle location
            location = data.get('location', {})
            if isinstance(location, dict):
                event['venue_name'] = location.get('name', '')
                event['venue_address'] = location.get('address', '')
                
            return event if event['event_name'] and event['start_date'] else None
            
        except Exception as e:
            logger.error(f"Error parsing schema event: {str(e)}")
            return None

class EventScraper:
    def __init__(self, max_retries=3, timeout=30, max_concurrent=5):
        self.extractor = EventExtractor()
        self.parser = ContentParser()
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.session = None
        self.processed_urls: Set[str] = set()
        self.rate_limit = asyncio.Semaphore(max_concurrent)
        self.events_cache: Dict[str, List[Event]] = {}
        
    async def init_session(self):
        """Initialize aiohttp session with proper settings"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=self.max_concurrent)
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=connector,
                headers={
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
            )

    async def close_session(self):
        """Clean up session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_with_retry(self, url: str) -> Optional[str]:
        """Fetch URL content with retries"""
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        await asyncio.sleep(retry_after)
                        continue
                    logger.warning(f"Attempt {attempt + 1}: Failed to fetch {url}, status: {response.status}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error fetching {url}: {str(e)}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        return None

    async def process_url(self, url: str) -> List[Event]:
        """Process a single URL and extract events"""
        async with self.rate_limit:
            if url in self.processed_urls:
                return self.events_cache.get(url, [])

            self.processed_urls.add(url)
            
            try:
                content = await self.fetch_with_retry(url)
                if not content:
                    return []

                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract structured data first
                structured_events = self.parser.extract_structured_data(soup)
                if structured_events:
                    events = [Event(**event, source_url=url) for event in structured_events]
                else:
                    # Use LLM extraction as fallback
                    events = await self.extractor.extract_event_data(content, url)

                # Enhance events with images and links
                images = self.parser.extract_images(soup, url)
                event_links = self.parser.extract_event_links(soup, url)
                
                for event in events:
                    if images:
                        event.image_url = images[0]
                    if event_links:
                        event.event_link = event_links[0]
                    else:
                        event.event_link = url

                self.events_cache[url] = events
                return events

            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                return []

    async def scrape_urls(self, urls: List[str]) -> pd.DataFrame:
        """Scrape multiple URLs and return results as DataFrame"""
        await self.init_session()
        try:
            tasks = [self.process_url(url) for url in urls]
            all_results = await asyncio.gather(*tasks)
            flattened_events = [event for url_results in all_results for event in url_results]
            
            # Convert to DataFrame
            df = pd.DataFrame([vars(event) for event in flattened_events])
            if not df.empty:
                df = df.drop_duplicates(subset=['event_name', 'start_date', 'venue_name'], keep='first')
                df = df.fillna('')
            return df
            
        finally:
            await self.close_session()

def main():
    st.set_page_config(page_title="Enhanced Event Scraper", layout="wide")
    st.title("Enhanced Event Scraper")

    with st.sidebar:
        st.header("Settings")
        max_retries = st.slider("Max retries per URL", 1, 5, 3)
        timeout = st.slider("Timeout (seconds)", 10, 60, 30)
        max_concurrent = st.slider("Max concurrent requests", 1, 10, 5)

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload URLs file (CSV or TXT)", type=["csv", "txt"])
    with col2:
        manual_url = st.text_input("Or enter a single URL:")

    if st.button("Start Scraping", type="primary"):
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
            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize scraper with settings
            scraper = EventScraper(
                max_retries=max_retries,
                timeout=timeout,
                max_concurrent=max_concurrent
            )

            try:
                # Create a progress container
                progress_container = st.container()
                with progress_container:
                    st.write("Scraping in progress...")
                    progress_bar = st.progress(0)
                    status = st.empty()

                # Run scraping
                df = asyncio.run(scraper.scrape_urls(urls))
                
                if not df.empty:
                    end_time = time.time()
                    duration = round(end_time - start_time, 2)
                    
                    # Success message and stats
                    st.success(f"✅ Scraping completed successfully in {duration} seconds!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Events", len(df))
                    with col2:
                        st.metric("Unique Venues", df['venue_name'].nunique())
                    with col3:
                        st.metric("Events with Images", df['image_url'].notna().sum())
                    with col4:
                        st.metric("URLs Processed", len(scraper.processed_urls))

                    # Data Quality Analysis
                    with st.expander("📊 Data Quality Analysis"):
                        # Calculate completion rates
                        completion_rates = pd.DataFrame({
                            'Field': df.columns,
                            'Completion Rate': [
                                f"{(df[col].notna().sum() / len(df) * 100):.1f}%" 
                                for col in df.columns
                            ]
                        })
                        st.write("Field Completion Rates:")
                        st.dataframe(completion_rates, hide_index=True)

                    # Filters
                    st.subheader("🔍 Filter Events")
                    col1, col2 = st.columns(2)
                    
                    filtered_df = df.copy()
                    
                    with col1:
                        if 'start_date' in df.columns and not df['start_date'].empty:
                            date_filter = st.date_input(
                                "Filter by Date Range",
                                value=None,
                                min_value=pd.to_datetime(df['start_date'].min()).date() if not df['start_date'].empty else None,
                                max_value=pd.to_datetime(df['start_date'].max()).date() if not df['start_date'].empty else None
                            )
                            if date_filter:
                                filtered_df = filtered_df[
                                    pd.to_datetime(filtered_df['start_date']).dt.date == date_filter
                                ]
                    
                    with col2:
                        if 'category' in df.columns and not df['category'].empty:
                            categories = sorted(df['category'].unique())
                            selected_categories = st.multiselect(
                                "Filter by Category",
                                options=categories
                            )
                            if selected_categories:
                                filtered_df = filtered_df[
                                    filtered_df['category'].isin(selected_categories)
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
                        hide_index=True
                    )

                    # Export options
                    st.subheader("💾 Export Data")
                    col1, col2 = st.columns(2)
                    
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
                                'bg_color': '#D9EAD3',
                                'border': 1
                            })
                            
                            for col_num, value in enumerate(filtered_df.columns.values):
                                worksheet.write(0, col_num, value, header_format)
                                width = max(
                                    len(value) + 2,
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
                else:
                    st.error("No events found. Please check the URLs and try again.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception("Scraping error")
        else:
            st.warning("Please provide at least one URL to scrape.")

if __name__ == "__main__":
    main()