
"""
AI News Curator Agent using LangGraph with Agentic Browser Tool
The agent uses browser automation to intelligently scrape news.smol.ai
"""

from typing import TypedDict, List, Literal
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, HttpUrl
import os
import random
import json
import re
from functools import partial
from bs4 import BeautifulSoup
import logging


_MAX_DAYS_TO_CHECK = 2  # Look back this many days for news articles
_MAX_TEXTS = 25
# ============================================================================
# STATE DEFINITION
# ============================================================================

tavily_client = TavilyClient(os.getenv("TAVILY_KEY"))

# Module logger: add a handler only if none exist to avoid duplicate logs in notebooks
logger = logging.getLogger(__name__)
# Ensure a single handler for this module and prevent propagation to root logger.
# In notebook environments previous handlers may already be attached (causing duplicates),
# so clear them and attach exactly one stream handler.
for h in list(logger.handlers):
    logger.removeHandler(h)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)

_THINKING_SUFFIX = "/no_think"
# _THINKING_SUFFIX = ""

def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags from model output"""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

_EXAMPLE_POST = """
Everyone’s talking about LLMs, but Sentence Transformers continues to quietly shine. LLMs are decoder only components, while Sentence transformers are the encoders.

It is one of the most practical and elegant libraries for embeddings.

Huge credit to Nils Reimers and Iryna Gurevych - the original authors of the hashtag#sbert paper, and maintainers , especially Tom Aarsen for keeping NLP simple, modular, and production-ready.

It also has adapter training enabled and it is a game changer. It is possible to train lightweight adapters (model sizes in kilobytes, huh!) per customer and just switch between them during hashtag#inference. No full retraining necessary. An example is shown below. 

Load the main model once and use adapters on the fly. Perfect for anyone building multi-tenant or domain-specific NLP systems.

hashtag#AI hashtag#NLP hashtag#MachineLearning hashtag#SentenceTransformers hashtag#Adapters hashtag#LLM"""

class News(BaseModel):
    """Each news article."""
    news_title: str = Field(description="A short title for the content usually within 10 words.")
    description: str = Field(description="A brief description or summary of the news article - max 25 words")
    date: str = Field(description="Date when the news was published")

class NewsList(BaseModel):
    """List of news articles."""
    articles: List[News] = Field(description="A list of the news articles")

class SearchQuery(BaseModel):
    """Search Query"""
    query: str = Field(description="Search query to use for web search")

class WebContent:
    """Web content with useful links
    """
    def __init__(self, summary, links_w_contents):
        self.summary = summary
        self.links_w_contents = links_w_contents

    # str = Field(description="Content summarized from the web")
    # links: List[Tuple[str, str]] = Field(description="Relevant URLs with content as (url, content) tuples")

class LinkedInPost(BaseModel):
    """LinkedIn Post"""
    post: str = Field(description="Generated LinkedIn post textual content")
    url: HttpUrl = Field(description="A selected web link that backs up the content")

class AgentState(TypedDict):
    """State that gets passed between nodes in the graph"""
    url: str
    llm: ChatOpenAI
    texts: List[str]
    raw_headlines: List[dict]  # List of {title, url, date, summary}
    top_stories: NewsList   # top 3-5 stories
    selected_story: News
    search_query: str
    retrieved_data: str        # Additional context from web search
    linkedin_post: LinkedInPost         # Final generated post


def selector_node(state: AgentState) -> AgentState:
    """
    Selects one article from the current set of accumulated articles
    """
    if state["top_stories"] is None:
        return []
    # print("In selector node:", state["top_stories"])

    # convert list of Pydantic objects to json
    json_out = [x.model_dump_json() for x in state["top_stories"].articles]
    # Use LLM to select the most interesting story
    prompt = f"""You are given a set of news articles. 
    Choose any one from the set that is the most interesting
    from a technical Linkedin user perspective. 
    Avoid news related to funding or governance. 
    
    Articles:
    {json_out}

    Return the selected article as a formatted object. {_THINKING_SUFFIX}"""
    output = state["llm"].with_structured_output(News).invoke(prompt)

    state["selected_story"] = output
    logger.info(f"Selected story: {state['selected_story']!r}")
    return state


def search_query_generator_node(state: AgentState) -> AgentState:
    """
    Search Query Generator

    """
    if state["selected_story"] is None:
        return ""

    prompt = f"""You are given a news article.
    Generate a suitable search query using which you can fetch more content using a search engine.
    Make sure the query focuses on only one topic. 
    The query should be informative enough to fetch some unique insights on the topic.
    The query should be precise, usually a maximum of 7 words. 

    Article: {state["selected_story"].model_dump_json()}

    Return the search query as a string. {_THINKING_SUFFIX}
    """
    search_query = state["llm"].with_structured_output(SearchQuery).invoke(prompt)
    state["search_query"] = search_query.query
    logger.info(f"Generated search query: {state['search_query']}")
    return state


def retrieval_node(state: AgentState) -> AgentState:
    """
    Retrieve contents from web articles
    """
    
    search_response = tavily_client.search(
        query=state["search_query"],
        time_range="month",
        max_results=5,
        include_answer="advanced",
        # include_domains=["news.smol.ai"]
    )

    if search_response:
        answer = search_response["answer"]
        url_contents = [(x["url"], x["content"]) for x in search_response["results"]] # if x["score"] >= 0.7]
        state["retrieved_data"] = WebContent(answer, url_contents)
    
    return state

def writer_node(state: AgentState) -> AgentState:
    """
    Generates the final LinkedIn post
    """
    if state["retrieved_data"] is None:
        logger.warning("Cannot generate Linkedin post without content")

    logger.info("✍️ Generating LinkedIn post...")

    # Prepare context
    summary = state["retrieved_data"].summary
    urls_with_content = " ".join([f"url: {x[0]}\n content: {x[1]}\n\n" for x in state["retrieved_data"].links_w_contents])

    # print(urls_with_content)
    # print(summary) 
    prompt = f"""Create an engaging post for LinkedIn based on the given contexts. 
    In addition, you also be given URL sources and corresponding summarized contents from the webpages - you can pick up some information from those if needed.

Context:
{summary}

Additional context:
{state["selected_story"].news_title} {state["selected_story"].description}


Links and contents: 
{urls_with_content[:4000]}

Requirements:
- Make it fun and interesting. Don't appear news like. Also keep it simple, not LLM like. 
- Rewrite, do not copy-paste.
- Use a conversational tone.
- Minimize use of emojis
- [optional] End with something thought-provoking
- Keep it around 250-300 words
- Use line breaks for readability
- Choose the URL from above which contains the most relevant content and add is as a reference in the post - for example, Read More: URL. 
- add some humor as well
- make it sound like a human wrote it
- remove em dashes and long dashes
- add hash tags in important places within the post as needed

Also, extract one URL out of the ones given that contains the most relevant information wrt the context. 

Make sure the post is relevant to the original search query: "{state['search_query']}"

Try to keep the tone similar to {_EXAMPLE_POST.strip()}

{_THINKING_SUFFIX}
    """

    try:
        response = state["llm"].with_structured_output(LinkedInPost).invoke(prompt)
        logger.debug("Generated content type: %s", type(response))
        state["linkedin_post"] = response
    except Exception as e:
        # Fallback: try to get raw response and parse manually
        logger.warning(f"Structured output failed, trying manual parsing: {e}")
        raw_response = state["llm"].invoke(prompt)

        # Strip thinking tags
        cleaned_content = strip_thinking_tags(raw_response.content)

        # Try to parse as JSON
        try:
            parsed = json.loads(cleaned_content)
            state["linkedin_post"] = LinkedInPost(**parsed)
        except json.JSONDecodeError:
            # If not valid JSON, assume the entire response is the post content
            # and try to extract a URL
            logger.warning("Could not parse as JSON, using raw content")
            urls = re.findall(r'https?://[^\s<>"]+', cleaned_content)
            default_url = urls[0] if urls else state["retrieved_data"].links_w_contents[0][0]
            state["linkedin_post"] = LinkedInPost(
                post=cleaned_content,
                url=default_url
            )

    return state


# ============================================================================
# BROWSER TOOLS
# ============================================================================


async def navigate_to_page(state: AgentState) -> AgentState:
    """
    launch browser to go to a URL and return the page text contents
    
    Returns:
        List of text contents extracted from the page

    """
    print("State of the dict:", state)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(state["url"], wait_until="networkidle", timeout=30000)
            
            # info = {
            #     "title": await page.title(),
            #     "url": page.url,
            #     "status": "success"
            # }
            content = await page.content()
            await browser.close()
            # return json.dumps(info)
            soup = BeautifulSoup(content, "html.parser")
            # get all texts from soup
      
            cleaned_texts = [x.text.strip() for x in soup.find_all("div", class_="group relative arrow-card")]
            # Return the cleaned texts (limit to max days to check)
            state["texts"] = cleaned_texts[:_MAX_DAYS_TO_CHECK]
    except Exception as e:
        logger.error("Navigating to page --> issues %s", str(e))
        return json.dumps({"status": "error", "error": str(e)})
    return state
    

async def extract_important_info(state: AgentState) -> AgentState:
        """
        Extract important info from text blocks
        
        Args:
            texts: Input texts from a web page
            topk: Max number of final important texts
            max_history: Max number of text blocks to consider
        
        Returns:
            List of topk texts extracted
        """
        if state["llm"] is None:
            raise ValueError("LLM is not available")
        
        if not state["texts"]:
            raise ValueError("No texts available")

        print(state)
        important_news_articles = []
        for text in state["texts"][:_MAX_DAYS_TO_CHECK]:
            prompt = f"""Gather some important articles available from the given text.
              Consider something important if it is related to AI and sounds like something promising.
              Given text: {text} {_THINKING_SUFFIX}"""
            news_from_text = await state["llm"].with_structured_output(NewsList).ainvoke(prompt)
            important_news_articles.extend(news_from_text.articles)
        
        if important_news_articles:
            state["top_stories"] = NewsList(articles=important_news_articles[:_MAX_TEXTS])
        return state



# # Replace your current extract_important_info tool with this factory function
# def create_extract_info_tool(llm: ChatOpenAI):
#     """Creates an extraction tool with the given LLM."""
    
#     @tool
#     async def extract_important_info(texts: List[str], topk=_MAX_TEXTS, max_history=_MAX_DAYS_TO_CHECK) -> NewsList:
#         """
#         Extract important info from text blocks
        
#         Args:
#             texts: Input texts from a web page
#             topk: Max number of final important texts
#             max_history: Max number of text blocks to consider
        
#         Returns:
#             List of topk texts extracted
#         """
#         if llm is None:
#             raise ValueError("LLM is not available")
            
#         important_news_articles = []
#         for text in texts[:max_history]:
#             prompt = f"""Gather some important articles available from the given text.
#               Consider something important if it is related to AI and sounds like something promising.
#               Given text: {text} {_THINKING_SUFFIX}"""
#             news_from_text = await llm.with_structured_output(NewsList).ainvoke(prompt)
#             important_news_articles.extend(news_from_text.articles)
        
#         if important_news_articles:
#             return NewsList(articles=important_news_articles[:topk])
#         return NewsList(articles=[])
        
#     return extract_important_info

# ============================================================================
# AGENT NODES
# ============================================================================

# async def agentic_scraper_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
#     """
#     Uses an LLM agent with tools to go to a URL and extract contents.
#     """
#     logger.info("🤖 Starting agentic scraper...")

#     extract_info_tool = create_extract_info_tool(llm)
#     tools = [navigate_to_page, extract_info_tool]
#     llm_with_tools = llm.bind_tools(tools)

#     # System message to guide the agent
#     system_msg = SystemMessage(content="""You will be given an URL.
# Go to the URL and then get all the contents. Once you collect all the contents from the page, extract important information into a list.""")
#     # Return your findings as a JSON array of articles with fields: title, date, url, summary. Do not make up content.
#     messages = [
#         system_msg,
#         HumanMessage(content=f"Extract texts from https://news.smol.ai/issues {_THINKING_SUFFIX}")
#     ]

#     # Agent loop - let it use tools iteratively
#     max_iterations = 2
#     for i in range(max_iterations):
#         logger.info("Iteration %d/%d", i + 1, max_iterations)

#         response = await llm_with_tools.ainvoke(messages)
#         messages.append(response)

#         # Check if the agent wants to use tools
#         if hasattr(response, 'tool_calls') and response.tool_calls:
#             for tool_call in response.tool_calls:
#                 tool_name = tool_call["name"]
#                 tool_args = tool_call["args"]

#                 logger.info("🔧 Using tool: %s", tool_name)
#                 logger.debug("Tool args keys: %s", list(tool_args.keys()))

#                 # Execute the tool
#                 if tool_name == "navigate_to_page":
#                     result = await navigate_to_page.ainvoke(tool_args)
#                     logger.debug("Tool: %s Tool result: %s", tool_name, result)
#                 elif tool_name == "extract_important_info":
#                     logger.debug("Tool args after update: %s", list(tool_args.keys()))
#                     result = await extract_info_tool.ainvoke(tool_args)
#                     logger.debug("Tool: %s Tool result: %s", tool_name, result)
#                 else:
#                     result = "Unknown tool"

#                 # Add tool result to messages
#                 messages.append(ToolMessage(
#                     content=str(result),
#                     tool_call_id=tool_call["id"],
#                     tool_name=tool_name,
#                     artifact=result
#                 ))
#         else:
#             # Agent is done using tools, extract final answer
#             break

#     # Parse the final response to extract articles
#     # Find the last ToolMessage from the extractor
#     final_response = ""
#     for msg in reversed(messages):
#         if isinstance(msg, ToolMessage) and getattr(msg, "tool_name", None) == "extract_important_info":
#             final_response = msg
#             state["top_stories"] = final_response
#             break

#     return state

# ============================================================================
# CONDITIONAL EDGES
# ============================================================================
# ============================================================================
# CONDITIONAL EDGES
# ============================================================================


def should_continue_research(state: AgentState) -> Literal["research", "curate"]:
    """
    Decides if we need more research or can proceed to curation
    """
    logger.debug("Deciding whether to continue research or curate...")
    # currently deterministic choice for simplicity; replace with real logic if needed
    decided = random.choice(["curate"])
    logger.debug("Randomly chosen %s", decided)
    return decided
# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent() -> StateGraph:
    # Initialize the graph
    workflow = StateGraph(AgentState)
    # workflow = StateGraph(dict)
    # Add nodes
    workflow.add_node("webpage_navigator", navigate_to_page)
    workflow.add_node("text_retriever", extract_important_info)
    workflow.add_node("selector", selector_node)
    workflow.add_node("query_generator", search_query_generator_node)
    workflow.add_node("retriever", retrieval_node)
    workflow.add_node("writer", writer_node)

    # # Define the flow
    workflow.set_entry_point("webpage_navigator")
    workflow.add_edge("webpage_navigator", "text_retriever")
    workflow.add_edge("text_retriever", "selector")
    workflow.add_edge("selector", "query_generator")
    workflow.add_edge("query_generator", "retriever")
    workflow.add_edge("retriever", "writer")

    # # instead of writer, can add a conditional edge to go back to scraper for more research
    # workflow.add_conditional_edges(
    #     "writer",
    #     should_continue_research,
    #     {"research": "selector", "curate": END},
    # )
    workflow.add_edge("writer", END)
    # Compile the graph
    app = workflow.compile()

    return app


