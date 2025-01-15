import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List, Dict
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

# Load environment variables
load_dotenv("var.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Fix for Windows asyncio compatibility
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Initialize OpenAI LLM
def init_llm():
    return ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Define the schema for event extraction
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
    "required": ["event_name", "venue_name", "start_date"]  # Reduced required fields for better coverage
}

class EventScraper:
    def __init__(self):
        self.llm = init_llm()
        
    def extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs from common patterns"""
        images = []
        # Look for images in various common patterns
        img_elements = soup.find_all('img', src=True)
        for img in img_elements:
            src = img.get('src', '')
            if any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                full_url = urljoin(base_url, src)
                images.append(full_url)
                
        # Look for background images in style attributes
        elements_with_style = soup.find_all(attrs={'style': True})
        for element in elements_with_style:
            style = element['style']
            if 'background-image' in style:
                url_match = re.search(r'url\(["\']?(.*?)["\']?\)', style)
                if url_match:
                    full_url = urljoin(base_url, url_match.group(1))
                    images.append(full_url)
                    
        return images

    def extract_event_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract event-specific links"""
        event_links = []
        # Look for links containing event-related keywords
        keywords = ['event', 'spectacle', 'concert', 'show', 'agenda', 'programmation']
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text().lower()
            if any(keyword in text or keyword in href.lower() for keyword in keywords):
                full_url = urljoin(base_url, href)
                event_links.append(full_url)
        return event_links

    async def scrape_url(self, url: str) -> List[Dict]:
        try:
            # Load HTML content
            loader = AsyncChromiumLoader([url])
            docs = await loader.aload()
            
            # Transform with BeautifulSoup
            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(
                docs,
                tags_to_extract=["div", "span", "a", "p", "h1", "h2", "h3", "h4", "img", "time", "article"]
            )
            
            # Parse with BeautifulSoup for additional extraction
            soup = BeautifulSoup(docs[0].page_content, 'html.parser')
            
            # Extract images and event links
            images = self.extract_images(soup, url)
            event_links = self.extract_event_links(soup, url)
            
            # Split content
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,  # Reduced chunk size for better precision
                chunk_overlap=100
            )
            splits = splitter.split_documents(docs_transformed)
            
            # Extract event data using LLM
            chain = create_extraction_chain(schema=SCHEMA, llm=self.llm)
            results = []
            
            for split in splits:
                # Enhance the content with structured data hints
                enhanced_content = f"""
                Page URL: {url}
                Available Images: {', '.join(images[:3])}  # Limit to first 3 images
                Event Links: {', '.join(event_links[:3])}  # Limit to first 3 links
                Content: {split.page_content}
                """
                
                extracted = chain.run(enhanced_content)
                if isinstance(extracted, list) and extracted:
                    for item in extracted:
                        # Add missing fields with available data
                        if not item.get('image_url') and images:
                            item['image_url'] = images[0]  # Use first available image
                            
                        if not item.get('event_link'):
                            matching_links = [link for link in event_links 
                                           if item.get('event_name', '').lower() in link.lower()]
                            if matching_links:
                                item['event_link'] = matching_links[0]
                            else:
                                item['event_link'] = url
                                
                        results.append(item)
            
            # Remove duplicates based on event name and start date
            unique_results = []
            seen = set()
            for item in results:
                key = (item.get('event_name', ''), item.get('start_date', ''))
                if key not in seen:
                    seen.add(key)
                    unique_results.append(item)
            
            return unique_results

        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
            return []

    async def scrape_urls(self, urls: List[str]) -> pd.DataFrame:
        tasks = [self.scrape_url(url) for url in urls]
        all_results = await asyncio.gather(*tasks)
        flattened_results = [event for url_results in all_results for event in url_results]
        
        # Convert to DataFrame and clean up
        df = pd.DataFrame(flattened_results)
        if not df.empty:
            # Fill missing values
            df = df.fillna({
                'end_date': df['start_date'],
                'end_time': df['start_time'],
                'category': 'Not specified',
                'description': 'No description available'
            })
            
            # Remove completely duplicate rows
            df = df.drop_duplicates()
            
            # Sort by start date
            try:
                df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
                df = df.sort_values('start_date')
            except:
                pass
                
        return df

def main():
    st.title("Event Data Scraper")
    
    # Add custom CSS for better visibility of error messages
    st.markdown("""
        <style>
        .stError {
            background-color: #ffebee;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader("Upload URLs file (CSV or TXT)", type=["csv", "txt"])

    # Manual URL input
    manual_url = st.text_input("Or enter a single URL:")
    
    # Add progress tracking
    progress_container = st.empty()

    scraper = EventScraper()

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
            total_urls = len(urls)
            progress_text = st.empty()
            
            with st.spinner('Scraping events...'):
                df = asyncio.run(scraper.scrape_urls(urls))

                if not df.empty:
                    st.success(f"Successfully scraped {len(df)} events from {total_urls} URLs!")
                    
                    # Display statistics
                    st.write("### Scraping Statistics")
                    st.write(f"- Total URLs processed: {total_urls}")
                    st.write(f"- Total events found: {len(df)}")
                    st.write(f"- URLs with successful scrapes: {len(df['event_link'].unique())}")
                    
                    # Display the dataframe
                    st.dataframe(df)

                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No events were successfully scraped. This might be due to:")
                    st.write("1. Website structure not compatible with current scraping patterns")
                    st.write("2. JavaScript-heavy websites that require additional processing")
                    st.write("3. Website blocking automated access")
        else:
            st.warning("Please provide URLs either through file upload or manual input.")

if __name__ == "__main__":
    main()