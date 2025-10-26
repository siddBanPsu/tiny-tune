
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
import json
from functools import partial
from bs4 import BeautifulSoup


_MAX_DAYS_TO_CHECK = 4  # Look back this many days for news articles
_MAX_TEXTS = 25
# ============================================================================
# STATE DEFINITION
# ============================================================================

tavily_client = TavilyClient(os.getenv("TAVILY_KEY"))

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
    raw_headlines: List[dict]  # List of {title, url, date, summary}
    top_stories: NewsList   # top 3-5 stories
    selected_story: News
    search_query: str
    retrieved_data: str        # Additional context from web search
    linkedin_post: LinkedInPost         # Final generated post


def selector_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Selects one article from the current set of accumulated articles
    """
    if state["top_stories"] is None:
        return []
    # print("In selector node:", state["top_stories"])

    news_list = [x[1] for x in state["top_stories"].artifact]

    out = []
    for item in news_list:
            out.extend(item)
    # convert list of Pydantic objects to json
    json_out = [x.json() for x in out]
    # Use LLM to select the most interesting story
    prompt = f"""You are given a set of news articles. Choose any one from the set that is the most interesting
    from a technical Linkedin user perspective. Avoid news related to funding or governance. 
    
    Articles:
    {json_out}

    Return the selected article as a formatted object. /no_think"""
    output = llm.with_structured_output(News).invoke(prompt)

    state["selected_story"] = output
    return state


def search_query_generator_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Based on the news article chosen, generate appropriate search query to fetch more information.
    """
    if state["selected_story"] is None:
        return ""

    prompt = f"""You are given a news article.
    Generate a suitable search query using which you can fetch more content using a search engine.
    Search queries should be precise, usually a maximum of 5-8 words. 

    Article: {state["selected_story"].model_dump_json()}

    Return the search query as a string. /no_think
    """
    search_query = llm.with_structured_output(SearchQuery).invoke(prompt)
    state["search_query"] = search_query.query
    return state


def retrieval_node(state: AgentState) -> AgentState:
    """
    Retrieve contents from web articles
    """
    
    search_response = tavily_client.search(
        query=state["search_query"],
        time_range="week",
        max_results=5,
        include_answer="advanced"
        # include_domains=["news.smol.ai"]
    )

    if search_response:
        answer = search_response["answer"]
        url_contents = [(x["url"], x["content"]) for x in search_response["results"] if x["score"] >= 0.7]
        state["retrieved_data"] = WebContent(answer, url_contents)
    
    return state

def writer_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Generates the final LinkedIn post
    """
    if state["retrieved_data"] is None:
        print("Cannot generate Linkedin post without content")

    print("✍️ Generating LinkedIn post...")

    # Prepare context
    summary = state["retrieved_data"].summary
    urls_with_content = " ".join([f"url: {x[0]}\n content: {x[1]}\n\n" for x in state["retrieved_data"].links_w_contents])

    # print(urls_with_content)
    # print(summary) 
    prompt = f"""Create an engaging post for LinkedIn based on the given context. 
    In addition, you also be given URL sources and corresponding summarized contents from the webpages - you can pick up some information from those if needed.

Context:
{summary}

Links and contents: 
{urls_with_content}

Requirements:
- Make it super fun and interesting. Add some personal touch - don't appear news like.
- Do not copy the content verbatim and add your own opinions and rewrite
- minimize use of emojis
- Avoid using long dashes or em dashes
- End with something thought-provoking
- Keep it under 200 words
- Use line breaks for readability
- Choose the URL from above which contains the most relevant content and add is as a reference in the post - for example, Read More: URL. 
- Generate some relevant hash tags in the end
- add some humor as well

Also, extract one URL out of the ones given that contains the most relevant information wrt the context. 

Generate the LinkedIn post now. /no_think"""

    response = llm.with_structured_output(LinkedInPost).invoke(prompt)
    print(f"Generated_content: \n {type(response)}")
    state["linkedin_post"] = response
    return state


# ============================================================================
# BROWSER TOOLS
# ============================================================================

@tool
async def navigate_to_page(url: str) -> List[str]:
    """
    launch browser to go to a URL and return the page text contents
    
    Args:
        url: The URL to navigate to
    
    Returns:
        List of text contents extracted from the page
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            info = {
                "title": await page.title(),
                "url": page.url,
                "status": "success"
            }
            content = await page.content()
            await browser.close()
            # return json.dumps(info)
            soup = BeautifulSoup(content, "html.parser")
            # get all texts from soup
      
            cleaned_texts = [x.text.strip() for x in soup.find_all("div", class_="group relative arrow-card")]
            return cleaned_texts[:_MAX_DAYS_TO_CHECK]
            cleaned_texts = "\n".join(x for x in cleaned_texts[:20])
            
            prompt = f"""Get the 10 most important contents from the given text.
                If there is nothing important, return empty list. Given text: {cleaned_texts}"""
            summary_list = await llm.with_structured_output(List[str]).ainvoke(prompt)
            print(summary_list)
            return summary_list
    except Exception as e:
        print("Navigating to page --> issues", str(e))
        return json.dumps({"status": "error", "error": str(e)})

# Replace your current extract_important_info tool with this factory function
def create_extract_info_tool(llm: ChatOpenAI):
    """Creates an extraction tool with the given LLM."""
    
    @tool
    async def extract_important_info(texts: List[str], topk=_MAX_TEXTS, max_history=_MAX_DAYS_TO_CHECK) -> NewsList:
        """
        Extract important info from text blocks
        
        Args:
            texts: Input texts from a web page
            topk: Max number of final important texts
            max_history: Max number of text blocks to consider
        
        Returns:
            List of topk texts extracted
        """
        if llm is None:
            raise ValueError("LLM is not available")
            
        important_news_articles = []
        for text in texts[:max_history]:
            prompt = f"""Gather some important articles available from the given text.
              Consider something important if it is related to AI and sounds like something promising.
              Given text: {text} /no_think"""
            news_from_text = await llm.with_structured_output(NewsList).ainvoke(prompt)
            important_news_articles.extend(news_from_text.articles)
        
        if important_news_articles:
            return NewsList(articles=important_news_articles[:topk])
        return NewsList(articles=[])
        
    return extract_important_info

# ============================================================================
# AGENT NODES
# ============================================================================

async def agentic_scraper_node(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """
    Uses an LLM agent with tools to go to a URL and extract contents.
    """
    print("🤖 Starting agentic scraper...")

    extract_info_tool = create_extract_info_tool(llm)
    
    tools = [navigate_to_page, extract_info_tool] 
    llm_with_tools = llm.bind_tools(tools)


    # System message to guide the agent
    system_msg = SystemMessage(content="""You will be given an URL.
Go to the URL and then get all the contents. Once you collect all the contents from the page, extract important information into a list.""")
# Return your findings as a JSON array of articles with fields: title, date, url, summary. Do not make up content.""")
    messages = [
    system_msg,
    HumanMessage(content="""Extract texts from https://news.smol.ai/issues /no_think""")
]
    
    # Agent loop - let it use tools iteratively
    max_iterations = 2
    for i in range(max_iterations):
        print(f"  Iteration {i+1}/{max_iterations}")
        
        response = await llm_with_tools.ainvoke(messages)
        # print(response)
        # print("  🤖 Agent response:", response.content[:200] + "..." if len(response.content) > 200 else response.content)
        messages.append(response)
        
        # print(response.tool_calls)
        # Check if the agent wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"  🔧 Using tool: {tool_name}")
                print(tool_args.keys())
                # Execute the tool
                if tool_name == "navigate_to_page":
                    result = await navigate_to_page.ainvoke(tool_args)
                    print(f"Tool: {tool_name} Tool result: {result}")
                elif tool_name == "extract_important_info":
                    # if "llm" not in tool_args:
                    print("Tool args after update:", tool_args.keys())
                    result = await extract_info_tool.ainvoke(tool_args)
                    print(f"Tool: {tool_name} Tool result: {result}")
                else:
                    result = "Unknown tool"
                
                # Add tool result to messages
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    tool_name=tool_name,
                    artifact=result
                ))
        else:
            # Agent is done using tools, extract final answer
            break
    
    # Parse the final response to extract articles
    # Find the last AIMessage (not ToolMessage)
    final_response = ""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.tool_name == "extract_important_info":
            final_response = msg
            state["top_stories"] = final_response
            break
    
    return state

# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_continue_research(state: AgentState) -> Literal["research", "curate"]:
    """
    Decides if we need more research or can proceed to curation
    """
    if state.get("iteration_count", 0) >= 1:
        return "curate"
    return "research"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent(llm: ChatOpenAI = None) -> StateGraph:
    """Creates and compiles the LangGraph agent"""
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("scraper", 
                      partial(agentic_scraper_node, llm=llm))
    workflow.add_node("selector", 
                      partial(selector_node, llm=llm))
    workflow.add_node("query_generator", 
                      partial(search_query_generator_node, llm=llm))
    workflow.add_node("retriever", retrieval_node)
    workflow.add_node("writer", partial(writer_node, llm=llm))
    
    # Define the flow
    workflow.set_entry_point("scraper")
    workflow.add_edge("scraper", "selector")
    workflow.add_edge("selector", "query_generator")
    workflow.add_edge("query_generator", "retriever")
    workflow.add_edge("retriever", "writer")

    workflow.add_edge("writer", END)

    # Compile the graph
    app = workflow.compile()
    
    return app


