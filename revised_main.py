import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
import logging
import time
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("var.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Fix for Windows asyncio compatibility
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def init_llm():
    return ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")  # Using 16k model for larger context

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
        "image_url": {"type": "string"},
        "price": {"type": "string"},  # Added price field
        "organizer": {"type": "string"}  # Added organizer field
    },
    "required": ["event_name", "event_link"]  # Reduced required fields to essential ones
}

class EventScraper:
    def __init__(self):
        self.llm = init_llm()
        self.session = None
        self.playwright = None
        self.browser = None
        self.results_cache = {}

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        if not self.playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
            )

    async def close_session(self):
        if self.session:
            await self.session.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def get_page_content(self, url: str) -> Optional[str]:
        try:
            context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = await context.new_page()
            
            # Set various timeouts
            page.set_default_timeout(30000)
            page.set_default_navigation_timeout(30000)
            
            # Enable JavaScript and wait for network idle
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Scroll to load lazy content
            await self._scroll_page(page)
            
            # Wait for dynamic content
            await page.wait_for_timeout(2000)
            
            # Get content after JavaScript execution
            content = await page.content()
            
            await context.close()
            return content
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    async def _scroll_page(self, page):
        """Scroll the page to trigger lazy loading"""
        try:
            await page.evaluate("""
                async () => {
                    await new Promise((resolve) => {
                        let totalHeight = 0;
                        const distance = 100;
                        const timer = setInterval(() => {
                            const scrollHeight = document.body.scrollHeight;
                            window.scrollBy(0, distance);
                            totalHeight += distance;
                            if(totalHeight >= scrollHeight) {
                                clearInterval(timer);
                                resolve();
                            }
                        }, 100);
                    });
                }
            """)
        except Exception as e:
            logger.warning(f"Scroll error: {str(e)}")

    async def extract_structured_data(self, html_content: str) -> List[Dict]:
        """Extract structured data from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for structured data
        structured_data = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get('@type') == 'Event':
                    structured_data.append(data)
            except:
                continue
                
        return structured_data

    async def scrape_url(self, url: str) -> List[Dict]:
        try:
            # Check cache
            if url in self.results_cache:
                return self.results_cache[url]

            content = await self.get_page_content(url)
            if not content:
                return []

            # First try to extract structured data
            structured_data = await self.extract_structured_data(content)
            if structured_data:
                results = self._convert_structured_data(structured_data, url)
            else:
                # Fall back to LLM extraction
                results = await self._extract_with_llm(content, url)

            # Cache results
            self.results_cache[url] = results
            return results

        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return []

    def _convert_structured_data(self, structured_data: List[Dict], url: str) -> List[Dict]:
        """Convert structured data to our schema format"""
        results = []
        for data in structured_data:
            event = {
                "event_name": data.get('name', ''),
                "venue_name": data.get('location', {}).get('name', ''),
                "venue_address": data.get('location', {}).get('address', ''),
                "start_date": data.get('startDate', ''),
                "end_date": data.get('endDate', ''),
                "category": data.get('eventAttendanceMode', ''),
                "event_link": url,
                "description": data.get('description', ''),
                "image_url": data.get('image', ''),
                "price": str(data.get('offers', {}).get('price', '')),
                "organizer": data.get('organizer', {}).get('name', '')
            }
            results.append(event)
        return results

    async def _extract_with_llm(self, content: str, url: str) -> List[Dict]:
        """Extract data using LLM when structured data is not available"""
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=4000,
            chunk_overlap=200
        )
        splits = splitter.split_text(content)

        chain = create_extraction_chain(schema=SCHEMA, llm=self.llm)
        results = []
        
        for split in splits:
            try:
                extracted = chain.run(split)
                if isinstance(extracted, list) and extracted:
                    for item in extracted:
                        item["event_link"] = url
                        results.append(item)
            except Exception as e:
                logger.error(f"LLM extraction error: {str(e)}")
                continue

        return results

    async def scrape_urls(self, urls: List[str]) -> pd.DataFrame:
        await self.init_session()
        try:
            tasks = [self.scrape_url(url) for url in urls]
            all_results = await asyncio.gather(*tasks)
            flattened_results = [event for url_results in all_results for event in url_results]
            return pd.DataFrame(flattened_results)
        finally:
            await self.close_session()

def main():
    st.title("Enhanced Event Data Scraper")
    st.markdown("""
    This scraper can handle:
    - Dynamic websites with JavaScript content
    - Structured data in JSON-LD format
    - Lazy-loaded content
    - Various event website formats
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload URLs file (CSV or TXT)", type=["csv", "txt"])

    # Manual URL input
    manual_url = st.text_input("Or enter a single URL:")

    # Add options
    with st.expander("Advanced Options"):
        wait_time = st.slider("Wait time for dynamic content (seconds)", 1, 10, 2)
        max_retries = st.slider("Maximum retries per URL", 1, 5, 3)

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
            # Validate URLs
            urls = [url for url in urls if urlparse(url).scheme in ['http', 'https']]
            
            if not urls:
                st.error("No valid URLs provided. Please check your input.")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            scraper = EventScraper()
            
            with st.spinner('Scraping events... This may take a while.'):
                df = asyncio.run(scraper.scrape_urls(urls))

                if not df.empty:
                    st.success(f"Successfully scraped {len(df)} events!")
                    
                    # Display preview
                    st.subheader("Preview of scraped data")
                    st.dataframe(df.head())
                    
                    # Statistics
                    st.subheader("Scraping Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Events", len(df))
                    with col2:
                        st.metric("Unique Venues", df['venue_name'].nunique())
                    with col3:
                        st.metric("Success Rate", f"{(len(df)/len(urls))*100:.1f}%")

                    # Download options
                    format_option = st.selectbox("Select download format:", ["CSV", "Excel"])
                    
                    if format_option == "CSV":
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        st.download_button(
                            label="Download Excel",
                            data=output.getvalue(),
                            file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.warning("No events were successfully scraped.")
        else:
            st.warning("Please provide URLs either through file upload or manual input.")

if __name__ == "__main__":
    main()