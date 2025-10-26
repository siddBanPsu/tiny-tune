import logging
import os
import re
import json
import requests
from random import sample
from datetime import datetime, timedelta
from typing import List, Optional
from tqdm import tqdm

from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kids_events.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)

# Constants
_MAX_RESULTS = 3

class EventFormat(BaseModel):
    """Each event."""
    event_title: str = Field(description="A short title for the event.")
    location: str = Field(description="Event location")
    date: str = Field(description="Event date and time")
    event_details: str = Field(description="Summary of what the event is about.")
    age_level: Optional[str] = Field(None, description="age group relevant for the event")
    url: Optional[str] = None

class EventList(BaseModel):
    """List of events"""
    event_list: List[EventFormat]

def initialize_clients():
    """Initialize Tavily and OpenAI clients."""
    logger.info("Initializing API clients")
    
    tavily_client = TavilyClient(os.getenv("TAVILY_KEY"))
    
    openai_client = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="qwen-lmstudio",
        temperature=0,
    )
    
    structured_llm = openai_client.with_structured_output(EventList)
    
    logger.info("API clients initialized successfully")
    return tavily_client, structured_llm

def get_weekend_dates_in_range(start_date):
    """Returns a list of all weekend dates within a given date range."""
    logger.info(f"Getting weekend dates starting from {start_date}")
    
    weekend_dates = []
    current_date = start_date
    while current_date <= start_date + timedelta(days=7):
        if current_date.weekday() in (5, 6):  # Saturday and Sunday
            weekend_dates.append(current_date.strftime(format="%d/%m/%Y"))
        current_date += timedelta(days=1)
    
    logger.info(f"Found {len(weekend_dates)} weekend dates: {weekend_dates}")
    return weekend_dates

def search_for_events(tavily_client):
    """Search for kid-friendly events using Tavily."""
    logger.info("Searching for kid-friendly events in Singapore")
    
    search_response = tavily_client.search(
        query="What are the kid friendly activities in Singapore happening in the coming weekend?",
        time_range="week",
        country="Singapore",
        max_results=_MAX_RESULTS
    )
    
    urls = [x["url"] for x in search_response["results"]]
    logger.info(f"Found {len(urls)} URLs to process: {urls}")
    
    return search_response, urls

def extract_and_process_events(tavily_client, structured_llm, urls, allowed_dates):
    """Extract content from URLs and process events."""
    logger.info("Extracting content from URLs")
    
    extracted_texts = tavily_client.extract(urls=urls)
    all_events = []

    for i, url in enumerate(urls):
        try:
            logger.info(f"Processing {i+1}/{len(urls)}: {url}")
            text_from_site = extracted_texts["results"][i]["raw_content"]

            chunked_texts = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=400,
            ).split_text(text_from_site)[:]
            
            logger.info(f"Created {len(chunked_texts)} text chunks for {url}")

            for j, text in enumerate(tqdm(chunked_texts, desc=f"Processing chunks from {url}")):
                try:
                    output = structured_llm.invoke(
                        f"""What are some kid-friendly activities happening in Singapore this weekend - only on these dates: {', '.join(allowed_dates)}. 
                        If no events match, return an empty list. Have a preference for events that are free or low cost and are outdoor.
                        Do not include events that are generic - like museums or zoos that are always open.
                        Answer based on the following content only: {text}"""
                    )
                    
                    if output.event_list:
                        for event in output.event_list:
                            event.url = url
                            all_events.append(event)
                        logger.info(f"Found {len(output.event_list)} events in chunk {j+1}")
                
                except Exception as e:
                    logger.warning(f"Error processing chunk {j+1} from {url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            continue

    logger.info(f"Total events extracted: {len(all_events)}")
    return all_events

def dedup_with_llm(events: List[EventFormat], structured_llm) -> EventList:
    """Deduplicate events using LLM."""
    logger.info(f"Deduplicating {len(events)} events using LLM")
    
    events_json = [e.model_dump() for e in events]
    sampled_events = sample(events_json, min(len(events_json), 30))
    
    logger.info(f"Sampled {len(sampled_events)} events for deduplication")
    
    prompt = f"""
    You are given a list of event objects in JSON. Some may be duplicates with slight wording differences.
    First, deduplicate them, keeping the most complete version of each unique event.
    Don't pick events that are too similar to each other.
    And then, choose ones that are super interesting, kids would love and unique and also do not pick events that are too generic.
    Suggest me only maximum of 7 such events. 
    Return only valid JSON: a list of event objects.
    
    Input:
    {json.dumps(sampled_events, indent=2)}
    """
    
    deduped = structured_llm.invoke(prompt)
    logger.info(f"Deduplication complete: {len(deduped.event_list)} unique events selected")
    
    return deduped

def events_to_markdown(events: List[EventFormat]) -> str:
    """Convert events to markdown format."""
    logger.info(f"Converting {len(events)} events to markdown")
    
    lines = []
    for idx, ev in enumerate(events, start=1):
        entry = (
            f"{idx}. {ev.event_title}*\n"
            f"Location: {ev.location}\n"
            f"Date: {ev.date}\n"
            f"Details: {ev.event_details}\n"
        )
        if ev.age_level:
            entry += f"Age group: {ev.age_level}\n"
        if ev.url:
            entry += f"[More Info]({ev.url})\n"
        lines.append(entry.strip())
    
    markdown_content = "Weekend events upcoming:\n\n" + "\n\n".join(lines)
    logger.info("Markdown conversion complete")
    
    return markdown_content

def escape_markdown_v2(text: str) -> str:
    """
    Escapes Telegram MarkdownV2 special characters.
    """
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


def send_events_to_telegram(content: str, bot_token: str, chat_id: str) -> dict:
    """Send events to Telegram chat."""
    logger.info("Sending events to Telegram")
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": escape_markdown_v2(content),
        "parse_mode": "MarkdownV2"
    }

    try:
        resp = requests.post(url, data=payload)
        resp.raise_for_status()
        logger.info("Successfully sent message to Telegram")
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send message to Telegram: {e}")
        raise

def main():
    """Main function to orchestrate the event processing pipeline."""
    logger.info("Starting kids events processing pipeline")
    
    try:
        # Initialize clients
        tavily_client, structured_llm = initialize_clients()
        
        # Get weekend dates
        allowed_dates = get_weekend_dates_in_range(datetime.now().date())
        
        # Search for events
        search_response, urls = search_for_events(tavily_client)
        
        # Extract and process events
        all_events = extract_and_process_events(tavily_client, structured_llm, urls, allowed_dates)
        
        if not all_events:
            logger.warning("No events found")
            return
        
        logger.info(f"Extracted {len(all_events)} events before deduplication")

        # Deduplicate events
        deduped_events = dedup_with_llm(all_events, structured_llm)
        
        # Convert to markdown
        markdown_output = events_to_markdown(deduped_events.event_list)
        
        # markdown_output = escape_markdown(markdown_output)
        
        # Send to Telegram
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if bot_token and chat_id:
            send_events_to_telegram(markdown_output, bot_token, chat_id)
        else:
            logger.warning("Telegram credentials not found, skipping notification")

        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        
        # Send error notification to Telegram
        # try:
        #     error_message = f"Kids events processing failed: {str(e)}"
        #     bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        #     chat_id = os.getenv("TELEGRAM_CHAT_ID")
            
        #     # if bot_token and chat_id:
        #     #     send_events_to_telegram(error_message, bot_token, chat_id)
        # except Exception as telegram_error:
        #     logger.error(f"Failed to send error notification: {telegram_error}")
        
        raise

if __name__ == "__main__":
    main()