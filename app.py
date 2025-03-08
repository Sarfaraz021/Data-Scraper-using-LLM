import streamlit as st
import pandas as pd
import asyncio
import time
from datetime import datetime
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from main import process_urls

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