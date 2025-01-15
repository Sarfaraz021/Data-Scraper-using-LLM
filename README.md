# Data-Scraper-using-LLM

A web scraping application that uses Language Learning Models (LLM) to extract and process data from websites.

## Description
This project combines the power of LLMs with web scraping capabilities to intelligently extract and analyze web content using Streamlit, LangChain, and Playwright.

## Methodology
![Web Scraping Methodology](web%20scrapping%20methodology%20digram.png)

## Prerequisites
- Python 3.11 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/Data-Scraper-using-LLM.git
```

2. Navigate to the project directory:
```
cd Data-Scraper-using-LLM
```

3. Install the required packages:
```
pip install streamlit langchain-openai langchain playwright beautifulsoup4 pandas
playwright install
pip install python-dotenv
```

## How to run the project

1. Set up environment variables:
Create a `.env` file in the project directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

2. Run the Streamlit application:
```
streamlit run app.py
```

## Environment setup details
- Ensure you have Python 3.11 or higher installed.
- Install the required packages as mentioned in the installation steps.
- Set up the environment variables as mentioned in the "How to run the project" section.