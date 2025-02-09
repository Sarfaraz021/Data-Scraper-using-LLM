import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.schema import Document
import os
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
from bs4 import BeautifulSoup, Tag
import re
from urllib.parse import urljoin, urlparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_anthropic import ChatAnthropic



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("var.env")
# os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

def init_llm():
    # return ChatOpenAI(temperature=0, model="gpt-4-turbo")
    return ChatAnthropic(model='claude-3-5-sonnet-20241022')

SCHEMA = {
    "properties": {
        "event_name": {"type": "string"},
        "venue_name": {"type": "string"},
        "venue_address": {"type": "string"},
        "start_date": {"type": "string"},
        "start_time": {"type": "string"},
        "end_date": {"type": "string"},
        "end_time": {"type": "string"},
        "category": {"type": "string"},
        "event_link": {"type": "string"},
        "description": {"type": "string"},
        "image_url": {"type": "string"}
    },
    "required": ["event_name", "start_date"]
}

class EventScraper:
    def __init__(self, max_retries=3, timeout=30, max_concurrent=5):
        self.llm = init_llm()
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.session = None
        self.processed_urls = set()
        self.rate_limit = asyncio.Semaphore(max_concurrent)

    async def init_session(self):
        if not self.session:
            connector = aiohttp.TCPConnector(limit=self.max_concurrent)
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=connector,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    def is_event_link(self, href: str, link_text: str) -> bool:
        """Enhanced event link detection"""
        href_lower = href.lower()
        text_lower = link_text.lower()
        
        # Common event-related keywords
        event_keywords = [
            'event', 'agenda', 'spectacle', 'exposition', 'concert',
            'theatre', 'festival', 'animation', 'conference', 'show',
            'performance', 'exhibition', 'program'
        ]
        
        # Date patterns
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        
        # Check for date in URL or link text
        has_date = bool(re.search(date_pattern, href) or re.search(date_pattern, text_lower))
        
        # Check for event keywords in URL or link text
        has_event_keyword = any(keyword in href_lower or keyword in text_lower for keyword in event_keywords)
        
        # Check if the link points to a specific event (usually longer URLs)
        is_specific_event = len(href.split('/')) > 4 and has_event_keyword
        
        return has_date or is_specific_event or has_event_keyword

    def find_event_container(self, soup: BeautifulSoup, url: str) -> Optional[Tag]:
        """Find the main container that holds event information"""
        # Common class/id patterns for event containers
        event_containers = soup.find_all(class_=re.compile(r'event|agenda|calendar|program', re.I))
        if not event_containers:
            event_containers = soup.find_all(id=re.compile(r'event|agenda|calendar|program', re.I))
        return event_containers[0] if event_containers else None

    def extract_event_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        event_links = set()
        container = self.find_event_container(soup, base_url)
        search_area = container if container else soup
        
        for link in search_area.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text(strip=True)
            
            # Skip empty or javascript links
            if not href or href.startswith(('javascript:', '#', 'mailto:')):
                continue
                
            absolute_url = urljoin(base_url, href)
            
            # Skip if already processed or external link
            if absolute_url in self.processed_urls or not urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                continue
            
            if self.is_event_link(href, link_text):
                event_links.add(absolute_url)
                
        return list(event_links)

    def extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        images = []
        container = self.find_event_container(soup, base_url)
        search_area = container if container else soup
        
        for img in search_area.find_all('img'):
            for attr in ['src', 'data-src', 'data-lazy-src', 'data-original']:
                src = img.get(attr)
                if src:
                    # Skip social media icons and small images
                    if 'icon' in src.lower() or 'logo' in src.lower() or 'social' in src.lower():
                        continue
                    absolute_url = urljoin(base_url, src)
                    images.append(absolute_url)
                    break
        
        return images

    async def process_page(self, url: str, content: str) -> List[Dict]:
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # First try to extract event data directly from HTML
            structured_data = self.extract_structured_data(soup)
            if structured_data:
                return structured_data
            
            # If no structured data, use LLM
            doc = Document(page_content=content, metadata={"source": url})
            
            chain = create_extraction_chain(schema=SCHEMA, llm=self.llm)
            results = []
            
            try:
                extracted = chain.run(doc.page_content)
                if isinstance(extracted, list) and extracted:
                    for item in extracted:
                        # Add images and links
                        images = self.extract_images(soup, url)
                        if images:
                            item["image_url"] = images[0]
                        
                        event_links = self.extract_event_links(soup, url)
                        if event_links:
                            item["event_link"] = event_links[0]
                        else:
                            item["event_link"] = url
                            
                        results.append(item)
            except Exception as e:
                logger.error(f"Error processing content from {url}: {str(e)}")

            return results
        except Exception as e:
            logger.error(f"Error in process_page for {url}: {str(e)}")
            return []

    def extract_structured_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract event data from structured HTML if available"""
        structured_data = []
        
        # Look for structured event data
        event_elements = soup.find_all(class_=re.compile(r'event|agenda-item|calendar-item', re.I))
        
        for element in event_elements:
            try:
                event_data = {}
                
                # Try to extract event name
                name_elem = element.find(class_=re.compile(r'title|name|heading', re.I))
                if name_elem:
                    event_data['event_name'] = name_elem.get_text(strip=True)
                
                # Try to extract date
                date_elem = element.find(class_=re.compile(r'date|time|when', re.I))
                if date_elem:
                    date_text = date_elem.get_text(strip=True)
                    # Add basic date parsing here
                    event_data['start_date'] = date_text
                
                if event_data.get('event_name') and event_data.get('start_date'):
                    structured_data.append(event_data)
            
            except Exception as e:
                logger.error(f"Error extracting structured data: {str(e)}")
                continue
        
        return structured_data

    async def scrape_url(self, url: str) -> List[Dict]:
        async with self.rate_limit:
            if url in self.processed_urls:
                return []

            self.processed_urls.add(url)
            try:
                content = await self.fetch_with_retry(url)
                if not content:
                    return []

                results = await self.process_page(url, content)
                
                # Scrape linked event pages
                soup = BeautifulSoup(content, 'html.parser')
                event_links = self.extract_event_links(soup, url)
                
                additional_results = []
                for event_link in event_links[:5]:
                    if event_link not in self.processed_urls:
                        event_content = await self.fetch_with_retry(event_link)
                        if event_content:
                            page_results = await self.process_page(event_link, event_content)
                            additional_results.extend(page_results)

                results.extend(additional_results)
                return results

            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
                return []

    async def fetch_with_retry(self, url: str) -> str:
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    logger.warning(f"Attempt {attempt + 1}: Failed to fetch {url}, status: {response.status}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}: Error fetching {url}: {str(e)}")
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        return ""

    async def scrape_urls(self, urls: List[str]) -> pd.DataFrame:
        await self.init_session()
        try:
            tasks = [self.scrape_url(url) for url in urls]
            all_results = await asyncio.gather(*tasks)
            flattened_results = [event for url_results in all_results for event in url_results]
        finally:
            await self.close_session()
        
        df = pd.DataFrame(flattened_results)
        if not df.empty:
            df = df.drop_duplicates(subset=['event_name', 'start_date'], keep='first')
            df = df.fillna('')
        return df

def main():
    st.title("Enhanced Event Data Scraper")

    uploaded_file = st.file_uploader("Upload URLs file (CSV or TXT)", type=["csv", "txt"])
    manual_url = st.text_input("Or enter a single URL:")

    with st.expander("Advanced Settings"):
        max_retries = st.slider("Max retries per URL", 1, 5, 3)
        timeout = st.slider("Timeout (seconds)", 10, 60, 30)
        max_concurrent = st.slider("Max concurrent requests", 1, 10, 5)

    if st.button("Start Scraping"):
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
            scraper = EventScraper(max_retries=max_retries, timeout=timeout, max_concurrent=max_concurrent)
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner('Scraping events...'):
                try:
                    df = asyncio.run(scraper.scrape_urls(urls))

                    if not df.empty:
                        end_time = time.time()
                        duration = round(end_time - start_time, 2)
                        
                        st.success(f"Successfully scraped {len(df)} events in {duration} seconds!")
                        st.dataframe(df)

                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                        st.subheader("Scraping Statistics")
                        st.write(f"Total URLs processed: {len(scraper.processed_urls)}")
                        st.write(f"Average time per URL: {round(duration/len(urls), 2)} seconds")
                    else:
                        st.warning("No events were successfully scraped.")
                except Exception as e:
                    st.error(f"An error occurred during scraping: {str(e)}")
                    logger.exception("Scraping error")
        else:
            st.warning("Please provide URLs either through file upload or manual input.")

if __name__ == "__main__":
    main()