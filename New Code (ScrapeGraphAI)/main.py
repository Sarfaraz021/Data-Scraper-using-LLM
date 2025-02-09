import streamlit as st
import pandas as pd
import json
from langchain.agents import initialize_agent, AgentType
from langchain_scrapegraph.tools import SmartScraperTool
from langchain_openai import ChatOpenAI
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv()

def init_scraper():
    tools = [SmartScraperTool()]
    agent = initialize_agent(
        tools=tools,
        llm=ChatOpenAI(temperature=0),
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

def scrape_events(url, fields):
    prompt = f"""
    Visit {url}, List all events from this webpage. For each event provide the following details:
    {', '.join(fields)}

    Make sure to must scrape image/ event banner url and event link.
    Return the data as a structured JSON array with each event as an object containing all available fields.
    """
    
    agent = init_scraper()
    return agent.run(prompt)

def main():
    st.set_page_config(page_title="Event Scraper", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 Advanced Event Scraper")
    
    # Sidebar for configurations
    with st.sidebar:
        st.header("⚙️ Settings")
        fields = st.multiselect(
            "Select fields to scrape",
            ["event_name", "venue_name", "venue_address", "start_date", "start_time",
             "end_date", "end_time", "category", "description", "price", "organizer",
             "image_url", "event_link"],
            default=["event_name", "start_date", "venue_name", "description", "image_url", "event_link"]
        )

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url_input = st.text_input("🔗 Enter URL to scrape", 
                                 placeholder="https://example.com/events")
        
    with col2:
        if st.button("🚀 Start Scraping", type="primary", disabled=not url_input):
            if url_input:
                try:
                    with st.spinner("🔍 Scraping events..."):
                        start_time = time.time()
                        
                        # Scrape events
                        result = scrape_events(url_input, fields)
                        
                        # Parse JSON response
                        events_data = json.loads(result)
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(events_data)
                        
                        end_time = time.time()
                        duration = round(end_time - start_time, 2)
                        
                        # Show results
                        st.success(f"✅ Successfully scraped {len(df)} events in {duration} seconds!")
                        
                        # Display data
                        st.subheader("📊 Scraped Events")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                        with col2:
                            excel_buffer = pd.ExcelWriter(f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", 
                                                        engine='xlsxwriter')
                            df.to_excel(excel_buffer, index=False)
                            excel_data = excel_buffer.save()
                            st.download_button(
                                label="📥 Download Excel",
                                data=excel_data,
                                file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
            else:
                st.warning("⚠️ Please enter a URL to scrape")

    # Footer
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Enter the URL of the events page you want to scrape
    2. Select the fields you want to extract in the sidebar
    3. Click 'Start Scraping' to begin
    4. Download the results in CSV or Excel format
    """)

if __name__ == "__main__":
    main()