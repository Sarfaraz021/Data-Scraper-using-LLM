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
    "required": [
        "event_name",
        "venue_name",
        "venue_address",
        "start_date",
        "start_time",
        "end_date",
        "end_time",
        "category",
        "event_link",
        "description",
        "image_url"
    ]
}

class EventScraper:
    def __init__(self):
        self.llm = init_llm()

    async def scrape_url(self, url: str) -> Dict:
        try:
            # Load HTML content
            loader = AsyncChromiumLoader([url])
            docs = await loader.aload()

            # Transform with BeautifulSoup
            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(
                docs,
                tags_to_extract=["div", "span", "a", "p", "h1", "h2", "h3", "img"]
            )

            # Split content
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=2000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(docs_transformed)

            # Extract event data using LLM
            chain = create_extraction_chain(schema=SCHEMA, llm=self.llm)
            results = []
            for split in splits[:3]:  # Process first 3 chunks to avoid token limits
                extracted = chain.run(split.page_content)
                if isinstance(extracted, list) and extracted:
                    results.extend(extracted)

            # Merge results if multiple chunks contained event data
            if results:
                final_result = results[0]
                final_result['event_link'] = url
                return final_result
            return None

        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
            return None

    async def scrape_urls(self, urls: List[str]) -> pd.DataFrame:
        tasks = [self.scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        results = [r for r in results if r is not None]
        return pd.DataFrame(results)

def main():
    st.title("Event Data Scraper")

    # File upload
    uploaded_file = st.file_uploader("Upload URLs file (CSV or TXT)", type=["csv", "txt"])

    # Manual URL input
    manual_url = st.text_input("Or enter a single URL:")

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
            with st.spinner('Scraping events... This may take a while.'):
                df = asyncio.run(scraper.scrape_urls(urls))

                if not df.empty:
                    st.success(f"Successfully scraped {len(df)} events!")
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
                    st.warning("No events were successfully scraped.")
        else:
            st.warning("Please provide URLs either through file upload or manual input.")

if __name__ == "__main__":
    main()
