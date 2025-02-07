# Built on Top of Dealey5.py - implementing multipage app instead of task_classification. 

import io
import os
from openai import OpenAI
from openai import AsyncOpenAI
from pprint import pprint
import time
from timeit import default_timer as timer
from langchain_community.document_loaders import FireCrawlLoader
from tqdm import tqdm
import logging
import sys
from section_dictionary import REPORT_SECTIONS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import streamlit as st

logging.info("Dealey application starting up")

client = OpenAI(api_key=st.secrets["openai_api_key"])
logging.info("OpenAI API key configured")

FirecrawlAPI_key =st.secrets["firecrawl_api_key"] 
logging.info("Firecrawl API key configured")

start_time = timer()
logging.info(f"Application timer started at: {start_time}")

async_client = AsyncOpenAI(api_key=st.secrets["openai_api_key"])
logging.info("Async OpenAI client initialized")

async def openai_call_wo_tools(model, message_history):
    try:
        logging.info(f"[openai_call_wo_tools] Starting call with model: {model}")
        response = await async_client.chat.completions.create(
            model=model,
            messages=message_history,
            temperature=0.8
        )
        response_message = response.choices[0].message
        response_message_content = response_message.content
        return response_message_content
    except Exception as e:
        logging.error(f"[openai_call_wo_tools] Error occurred: {str(e)}", exc_info=True)
        return None

def openai_w_tools(messages, tools, tool_choice, model):
    try:
        logging.info(f"[openai_w_tools] Starting call with model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        logging.error(f"[openai_w_tools] Unable to generate ChatCompletion response: {str(e)}", exc_info=True)
        return e

tool_set = [
        {
            "type": "function",
            "function": {
                "name": "google_general_search",
                "description": "Performs a Google search, returns the top 5 results, and scrapes the content of each result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The query to search. E.g. What is the 2024 revenue of company Salesforce?",
                        },
                        "gl": {
                            "type": "string",
                            "description": "A two letter country code to localise search reults. E.g. 'us' for USA, 'in' for India"
                        }
                    },
                    "required": ["search_query"],
                },
            },
        }]
logging.info(f"Tool set defined with {len(tool_set)} tools")

quests
import json

def google_general_search(search_query, gl):
    logging.info(f"Starting google_general_search with query: {search_query} and location: {gl}")
    search_query.strip("''").strip()
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": "google",
        "q": search_query,
        "api_key": st.secrets["search_api_key"],
        "gl": gl,
        "num": 3
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        logging.info(f"Received response from search API with status code: {response.status_code}")

        results = response.text
        results_json = json.loads(results)
        list_of_results = results_json.get("organic_results", [])
        
        google_list_of_results = []
        for result in list_of_results:
            link_url = result.get('link', "URL not retrieved.")
            link_title = result.get('title', "TITLE not retrieved.")
            link_snippet = result.get('snippet', "Snippet not retrieved.")
            
            link_set = {
                "link_title": link_title,
                "link_url": link_url,
                "link_snippet": link_snippet
            }
            google_list_of_results.append(link_set)

        logging.info(f"google_general_search completed successfully with {len(google_list_of_results)} results")
        return google_list_of_results

    except requests.RequestException as e:
        logging.error(f"HTTP Request failed: {str(e)}")
        return f"Search results failed. {str(e)}"
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode failed: {str(e)}")
        return f"Failed to decode JSON response. {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return f"An unexpected error occurred. {str(e)}"

async def task_planning(task_classification_label, user_query, tool_set, output_format, task_context_message=''):
    logging.info(f"[task_planning] Starting task planning for {task_classification_label} - {user_query}")
    task_planning_system_prompt = f'''You are an experienced VC Analyst helping your manager, a principal partner at a VC fund at doing tasks.
  To do this, here is all the information that will be shared with you:
  1. Task_classification_label: This is a label for the specific task you are currently working on.
  2. User_query: The task request statement shared by your manager.
  3. Expected_output_format: The structure and sections of the final report to be generated.

  Your main goal is to create a step by step plan of gathering all relevant information to complete this task. To do this you have access to the following tools:

  ###TOOLS AVAILABLE
  {tool_set}

  Your final output should always be in the format specified below.

  Here are some examples:
  USER: "
  ###TASK CLASSIFICATION LABEL
  ###create_macro_industry_reports

  ###USER QUERY
  Healthcare AI in India

  ##EXPECTED OUTPUT FORMAT
    [Format for macro industry report...]"
  AGENT: "<tool-calls>
          <tool-call>
            <tool-name>google_general_search</tool-name>
            <parameters>
              <search_query>recent funding deals in healthcare AI startups in India</search_query>
              <gl>in</gl>
            </parameters>
          </tool-call>
          <tool-call>
            <tool-name>google_general_search</tool-name>
            <parameters>
              <search_query>top funded healthcare AI startups in India in the last 6 months</search_query>
              <gl>in</gl>
            </parameters>
          </tool-call>
          <tool-call>
            <tool-name>google_general_search</tool-name>
            <parameters>
              <search_query>leading investors in Indian healthcare AI startups</search_query>
              <gl>in</gl>
            </parameters>
          </tool-call>
          <tool-call>
            <tool-name>google_general_search</tool-name>
            <parameters>
              <search_query>healthcare AI startup funding trends in India</search_query>
              <gl>in</gl>
            </parameters>
          </tool-call>
          <tool-call>
            <tool-name>google_general_search</tool-name>
            <parameters>
              <search_query>Indian healthcare AI market size and growth rate</search_query>
              <gl>in</gl>
            </parameters>
          </tool-call>
          <tool-call>
            <tool-name>google_general_search</tool-name>
            <parameters>
              <search_query>competitive landscape of healthcare AI in India</search_query>
              <gl>in</gl>
            </parameters>
          </tool-call>
          <tool-call>
            <tool-name>google_general_search</tool-name>
            <parameters>
              <search_query>regulatory environment for healthcare AI in India</search_query>
              <gl>in</gl>
            </parameters>
          </tool-call>
          <tool-call>
            <tool-name>google_general_search</tool-name>
            <parameters>
              <search_query>consumer behavior towards healthcare AI in India</search_query>
              <gl>in</gl>
            </parameters>
          </tool-call>
        </tool-calls>"
    
    USER: "###TASK CLASSIFICATION LABEL
    ###create_company_profiles

    ###USER QUERY
    Google

    ###EXPECTED OUTPUT FORMAT
    [Format for company profile...]
    AGENT: "
    <tool-calls> 
    <tool-call> 
        <tool-name>google_general_search</tool-name>
        <parameters>
          <search_query>Google company overview: industry, founding date, headquarters, brief history</search_query> 
          <gl>us</gl> 
        </parameters> 
    </tool-call>
    <tool-call> 
        <tool-name>google_general_search</tool-name>
        <parameters> 
            <search_query>Google core products and services: description, target market, unique selling proposition</search_query>
            <gl>us</gl>
        </parameters>
    </tool-call>
    <tool-call>
        <tool-name>google_general_search</tool-name>
        <parameters>
            <search_query>Google financials: revenue model, current revenue, profitability, funding history</search_query>
            <gl>us</gl>
        </parameters>
    </tool-call>
    <tool-call> 
        <tool-name>google_general_search</tool-name>
        <parameters> <search_query>Google leadership: founders background, key executives, board members</search_query> <gl>us</gl> </parameters> </tool-call> <tool-call> <tool-name>google_general_search</tool-name> <parameters> <search_query>Google team: notable hires, team size, growth rate</search_query> <gl>us</gl> </parameters> </tool-call> <tool-call> <tool-name>google_general_search</tool-name> <parameters> <search_query>Google market position: key metrics, user base, customer growth rate</search_query> <gl>us</gl> </parameters> </tool-call> <tool-call> <tool-name>google_general_search</tool-name> <parameters> <search_query>Google partnerships and clients: notable collaborations and business relationships</search_query> <gl>us</gl> </parameters> </tool-call> <tool-call> <tool-name>google_general_search</tool-name> <parameters> <search_query>Google market analysis: total addressable market (TAM), key competitors, recent trends</search_query> <gl>us</gl> </parameters> </tool-call> <tool-call> <tool-name>google_general_search</tool-name> <parameters> <search_query>Google recent developments: latest news, milestones, product launches 2024</search_query> <gl>us</gl> </parameters> </tool-call> <tool-call> <tool-name>google_general_search</tool-name> <parameters> <search_query>Google future outlook: strategic initiatives, upcoming expansions, industry awards</search_query> <gl>us</gl> </parameters> </tool-call> </tool-calls>"

  Make sure to include the 'gl' parameter in all google_general_search tool calls, adjusting the country code as appropriate for the company or industry being researched.
  Your research plan should cover all sections specified in the Expected_output_format. Ensure that your tool calls are comprehensive and will gather all necessary information to complete each section of the final report.
  Make sure that your output is only in the XML format specified in the examples above.'''
    logging.info(f"task_planning_system_prompt initialised")
    task_planning_task_prompt = f'''
  ###TASK CLASSIFICATION LABEL
  {task_classification_label}

  ###USER QUERY
  {user_query}

  ###ALL TASK CONTEXT
  {task_context_message}

  ####EXPECTED OUTPUT FORMAT
  {output_format}'''
    logging.info(f"task_planning_task_prompt initialised")
    task_planning_chat_history = [{"role": "system",
                                 "content": task_planning_system_prompt},
                                {"role": "user",
                                 "content": task_planning_task_prompt}]
    logging.info("Calling OpenAI for task planning.")
    task_planning = await openai_call_wo_tools("gpt-4o", task_planning_chat_history)
    logging.info("Task planning OpenAI Call complete.")
    return task_planning

def run_task_planning(task_classification_label, user_query, tool_set, output_format):
    logging.info(f"[run_task_planning] Running task planning synchronously for {task_classification_label} - {user_query}")
    result = asyncio.run(task_planning(task_classification_label, user_query, tool_set, output_format))
    logging.info(f"[run_task_planning] Running task planning synchronously for {task_classification_label} - {user_query}")
    return result

def plan_executor(task_plan, tool_set):
    logging.info("Starting plan execution")
    plan_executor_system_prompt = f'''You are a function executor LLM, with access to the following tools:

    ###LIST OF TOOLS
    {tool_set}

    Your main task is to execute the plan shared by the user.

    This plan will include a series of tool calls with necessary parameters presented in the XML format.'''

    task_execution_chat_history = [{"role": "system",
                                    "content": plan_executor_system_prompt},
                                   {"role": "user",
                                    "content": task_plan}]

    try:
        tool_call_response = openai_w_tools(task_execution_chat_history, tool_set, "auto", "gpt-3.5-turbo")
        logging.info("Tool call response received")
        logging.info(f"Tool_call_response_received: {tool_call_response}")
        tool_calls = tool_call_response.choices[0].message.tool_calls
    except Exception as e:
        logging.error(f"Error during tool call: {str(e)}")
        return []

    execution_results = []
    if tool_calls:
        logging.info(f"Executing {len(tool_calls)} tool calls")
        for tool_call in tool_calls:
            logging.info(f"Executing tool call {tool_call.id}")
            tool_call_id = tool_call.id
            tool_function_name = tool_call.function.name
            tool_query_string = eval(tool_call.function.arguments)['search_query']
            tool_location_string = eval(tool_call.function.arguments)['gl']

            # Step 3: Call the function and retrieve results. Append the results to the messages list.
            try:
                if tool_function_name == 'google_general_search':
                    logging.info(f"Using '{tool_function_name}' with '{tool_query_string}'.")
                    results = google_general_search(tool_query_string, tool_location_string)
                    if not isinstance(results, str):
                        logging.info(f"Tool call successfully done.")
                        execution_results.append(results)
                        logging.info(f"Results successfully appended.")
                    else:
                        logging.error(f"Tool call failed with response: {results}")
                        sys.exit(1)
                else:
                    logging.error(f"Error: function {tool_function_name} does not exist")
            except Exception as e:
                logging.error(f"Error executing function {tool_function_name}: {str(e)}")
                continue
            time.sleep(1)
    else:
        logging.warning("No tool calls identified by the model")
        if tool_call_response.choices[0].message.content:
            logging.info(f"Message content: {tool_call_response.choices[0].message.content}")

    logging.info("Plan execution completed")
    return execution_results


import aiohttp
import asyncio
import certifi
import ssl

FirecrawlAPI_key = st.secrets["firecrawl_api_key"]

async def fetch_data(session, url):
    json_payload = {
        'url': url,
        'pageOptions': {
            'onlyMainContent': True
        }
    }

    try:
        logging.info(f"Starting fetch for URL: {url}")
        async with session.post('https://api.firecrawl.dev/v0/scrape',
                                headers={'Authorization': f'Bearer {FirecrawlAPI_key}'},
                                json=json_payload,
                                ssl=ssl.create_default_context(cafile=certifi.where())) as response:
            if response.status == 200:
                logging.info(f"Successfully fetched data for URL: {url}")
                return await response.json()
            else:
                error_text = await response.text()
                return {'error': f'HTTP error {response.status}: {error_text}'}
    except aiohttp.ClientConnectorError as e:
        logging.error(f"Client connector error for URL {url}: {str(e)}")
        return {'error': f'Client connector error: {str(e)}'}
    except aiohttp.ClientSSLError as e:
      logging.error(f"SSL error for URL {url}: {str(e)}")
      return {'error': f'SSL error: {str(e)}'}
    except Exception as e:
        logging.error(f"An exception occurred for URL {url}: {str(e)}")
        return {'error': f'An exception occurred: {str(e)}'}

from tqdm.asyncio import tqdm 

async def research_workspace(list_of_links):
    logging.info(f"Starting research_workspace process with {len(list_of_links)} link entries")
    unique_links = set()
    url_to_items = {}
    
    # Collect unique links and associate them with items
    for entry in list_of_links:
        for item in entry:
            if not isinstance(item, dict):
                logging.error(f"Item is not a dictionary: {item}")
                continue
            if 'link_url' not in item:
                logging.error(f"Item missing 'link_url': {item}")
                continue
            if item['link_url'] != "LINK NOT RETRIEVED":
                url = item['link_url']
                unique_links.add(url)
                if url not in url_to_items:
                    url_to_items[url] = []
                url_to_items[url].append(item)
    
    logging.info(f"Number of unique links to scrape: {len(unique_links)}")
    
    # Scrape unique links with batching
    batch_size = 10  # Number of requests per batch
    delay_between_batches = 60  # Delay between batches in seconds
    
    async with aiohttp.ClientSession() as session:
        tasks = list(unique_links)
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_tasks = [fetch_data(session, url) for url in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
            
            # Wait before the next batch if there are more batches to process
            if i + batch_size < len(tasks):
                logging.info(f"Waiting {delay_between_batches} seconds before processing the next batch.")
                await asyncio.sleep(delay_between_batches)
    
    # Process results and update items
    for result in results:
        # Ensure that the result is not an Exception
        if isinstance(result, Exception):
            logging.error(f"Task failed with exception: {result}")
            continue
    
        if 'error' not in result:
            # Retrieve data and metadata
            data = result.get('data', {})
            
            metadata = data.get('metadata', {})
            
        
            # Ensure metadata is a dictionary
            if not isinstance(metadata, dict):
                logging.error(f"Metadata is not a dictionary.")
                continue
        
            url = metadata.get('sourceURL')
            if not url:
                logging.error(f"Result missing URL in metadata.")
                continue
        
            logging.info(f"Processing result for URL: {url}")
        
            if url in url_to_items:
                page_content = data.get('markdown', 'No content found')
                
                logging.info(f"Extracted page content.")
                for item in url_to_items[url]:
                    item['page_text'] = page_content
                logging.info(f"Updated {len(url_to_items[url])} items with page content")
            else:
                logging.warning(f"URL {url} not found in url_to_items")
        else:
            error_message = result.get('error', 'Unknown error')
            for item in url_to_items.get(result.get('url'), []):
                item['page_text'] = f"Error: {error_message}"
            logging.error(f"Error fetching data for URL in result: {error_message}")
    logging.info(f"Completed research_workspace process. Processed {len(unique_links)} unique links.")
    return list_of_links

from llama_index.core import VectorStoreIndex, Document, get_response_synthesizer
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine trieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore

openai.api_key = st.secrets["openai_api_key"]
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.embed_model = OpenAIEmbedding()

def chunking_research_ouptut(list_of_links):
    logging.info("Starting chunking_research_output function")
    documents = []
    chunk_size = 1000
    overlap = 200
    for entry in list_of_links: 
        for item in entry:
            page_content = item.get("page_text", "")
            if page_content:
                chunks_page_content = [page_content[i:i+chunk_size] for i in range(0, len(page_content), chunk_size-overlap)]
                logging.debug(f"Created {len(chunks_page_content)} chunks for page: {item.get('link_url', 'url_na')}")
                for chunk in chunks_page_content:
                    chunk_metadata = {"link_url": item.get("link_url", "url_na"),
                                      "link_title": item.get("link_title", "title_na")}
                    doc = Document(text=chunk, metadata=chunk_metadata)
                    documents.append(doc)
            else:
                logging.warning(f"Empty page content for URL: {item.get('link_url', 'url_na')}")
    logging.info(f"Chunking complete. Created {len(documents)} documents")
    return documents

def query_engine_generation(documents):
    logging.info("Starting query engine generation")
    logging.debug(f"Generating index from {len(documents)} documents")
    index = VectorStoreIndex.from_documents(documents)
    logging.debug("Creating VectorIndexRetriever")
    retriever_1 = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        vector_store_info='''Index of documents containing "text" which are chunks from webpages and "metadata" which includes the link_url and link_title for the webpage that the chunk is from.'''
    )
    # assemble query engine
    logging.debug("Assembling RetrieverQueryEngine")
    query_engine_1 = RetrieverQueryEngine(
        retriever=retriever_1
    )
    logging.info("Query engine generated successfully")
    return query_engine_1

def query_engine_running(query_engine, user_query):
    logging.info(f"Running query engine with user query: {user_query}...")
    try:
        response_2 = query_engine.query(user_query)
        logging.debug("Query executed successfully")
        response_format = {"response": response_2.response, 
                       "response_sources": response_2.source_nodes[0].node.get_metadata_str()}
        logging.info("Query response formatted")
        return response_format
    except Exception as e:
        logging.error(f"Error running query engine: {str(e)}")
        raise

from typing import List, Dict, Tuple

async def query_generation(user_query: str, list_of_sections: List[Dict], task_context_required='') -> List[Dict]:
    logging.info(f"Starting query generation for user query: {user_query}...")
    logging.info(f"list_of_sections received in query_generation: {list_of_sections}")
    logging.info(f"Type of list_of_sections: {type(list_of_sections)}")
    logging.info(f"Length of list_of_sections: {len(list_of_sections)}")
    logging.info(f"Task context required: {task_context_required}")

    if not list_of_sections:
        logging.error("list_of_sections is empty. Cannot generate queries.")
        return []
    
    async def generate_queries_for_section(section: Dict) -> Dict:
        logging.debug(f"Generating queries for section: {section['name']}")
        logging.debug(f"Section details: {section}")
        query_generation_system_prompt = f'''You are an experienced VC Analyst creating the final output of the following task for a user, a principal partner at a VC fund. Here is all the information available to you:

###USER QUERY
{user_query}

###ALL TASK CONTEXT
{task_context_required}

###SECTION INFORMATION
Name: {section['name']}
Description: {section['description']}
Best Practices:
{(section['best_practices'])}

Your current task is to generate a list of queries to search for information related to this section from a research workspace which includes a variety of resources for the task.

Your final output should include nothing but the list of queries in the specified format below.

####OUTPUT FORMAT

<QUERIES>
<QUERY>"enter query text"</QUERY>
<QUERY>"enter query 2 text"</QUERY>
</QUERIES>'''
        try:
            query_generation_chat_history = [{"role": "system", "content": query_generation_system_prompt}]
            query_generation = await openai_call_wo_tools("gpt-4o", query_generation_chat_history)
            logging.info(f"Queries generated successfully for section: {section['name']}")
            logging.debug(f"Generated queries: {query_generation}")
        except Exception as e:
            logging.error(f"Error generating queries for section {section['name']}: {str(e)}")
            raise
        
        return {
            "section_key": section['key'],
            "section_name": section['name'],
            "queries": query_generation
        }
    logging.info(f"Generating queries for {len(list_of_sections)} sections")

    tasks = []
    for section in list_of_sections:
        try:
            task = generate_queries_for_section(section)
            tasks.append(task)
        except Exception as e:
            logging.error(f"Error creating task for section {section.get('name', 'Unknown')}: {str(e)}")
    
    try:
        results = await asyncio.gather(*tasks)
        logging.info(f"Query generation completed for all sections. Number of results: {len(results)}")
        for result in results:
            logging.debug(f"Result for section {result['section_name']}: {result['queries']}")
        return results
    except Exception as e:
        logging.error(f"Error during query generation: {str(e)}")
        raise



async def information_retrieval(section_queries: List[Dict], query_engine) -> List[Dict]:
    logging.info("Starting information retrieval process")
    async def process_query(query: str) -> Dict:
        logging.debug(f"Processing query: {query}...")
        # Use asyncio.to_thread to run the synchronous function in a separate thread
        try:
            # Use asyncio.to_thread to run the synchronous function in a separate thread
            query_response = await asyncio.to_thread(query_engine_running, query_engine, query)
            logging.debug("Query processed successfully")
            return {"query": query, "answer": query_response}
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {"query": query, "answer": f"Error: {str(e)}"}

    async def process_section(section: Dict) -> Dict:
        logging.info(f"Processing section: {section['section_name']}")
        queries = re.findall(r'<QUERY>\s*"([^"]+)"\s*</QUERY>', section['queries'])
        logging.debug(f"Found {len(queries)} queries in section")
        tasks = [process_query(query) for query in queries]
        answers = await asyncio.gather(*tasks)
        logging.info(f"Completed processing section: {section['section_name']}")
        return {
            "section_key": section['section_key'],
            "section_name": section['section_name'],
            "answers": answers
        }

    tasks = [process_section(section) for section in section_queries]
    try:
        results = await asyncio.gather(*tasks)
        logging.info("Information retrieval process completed successfully")
        return results
    except Exception as e:
        logging.error(f"Error in information retrieval process: {str(e)}")
        raise

# Your original synchronous query_engine_running function
def query_engine_running(query_engine, user_query):
    logging.debug(f"Running query engine with query: {user_query[:50]}...")
    response_2 = query_engine.query(user_query)
    try:
        response_2 = query_engine.query(user_query)
        response_format = {
            "response": response_2.response,
            "response_sources": response_2.source_nodes[0].node.get_metadata_str()
        }
        logging.debug("Query engine ran successfully")
        return response_format
    except Exception as e:
        logging.error(f"Error running query engine: {str(e)}")
        raise

import asyncio

async def output_generation(user_query: str, task_context_required: str, section_results: List[Dict], output_format: str):
    logging.info("Starting output generation process")
    async def generate_section(section: Dict) -> Tuple[str, str]:
        logging.info(f"Generating content for section: {section['section_name']}")
        section_prompt = f'''You are an experienced VC Analyst creating a section of a report for a user, a principal partner at a VC fund. Here is all the information available to you:

        ###USER QUERY
        {user_query}

        ###ALL TASK CONTEXT
        {task_context_required}

        ###SECTION NAME
        {section['section_name']}

        ###SECTION RESEARCH
        {json.dumps(section['answers'], indent=2)}

        ###OUTPUT FORMAT
        {output_format}

        Your task is to generate the content for the "{section['section_name']}" section of the report.
        Follow the best practices provided and use the research information to create a detailed and informative section.
        Include relevant citations where appropriate.
        Your output should be in the specified format and ready to be inserted directly into the final report.'''

        section_chat_history = [{"role": "system", "content": section_prompt}]
        try:
            section_content = await openai_call_wo_tools("gpt-4o", section_chat_history)
            logging.info(f"Content generated successfully for section: {section['section_name']}")
            return (section['section_name'], f"\n\n{section_content}\n\n")
        except Exception as e:
            logging.error(f"Error generating content for section {section['section_name']}: {str(e)}")
            return (section['section_name'], f"\n\nError generating content: {str(e)}\n\n")

    # Create tasks for all sections
    tasks = [generate_section(section) for section in section_results]
    
    # Use as_completed to yield results as they finish
    for completed_task in asyncio.as_completed(tasks):
        try:
            section_name, section_content = await completed_task
            logging.debug(f"Section completed: {section_name}")
            yield section_name, section_content
        except Exception as e:
            logging.error(f"Error processing completed task: {str(e)}")

def run_output_generation(user_query, task_context_required, section_results, output_format): 
    logging.info("Starting run_output_generation")
    output_placeholder = st.empty()
    full_output = ""

    async def process_sections():
        nonlocal full_output
        async for section_name, section_content in output_generation(user_query, task_context_required, section_results, output_format):
            full_output += section_content
            output_placeholder.markdown(full_output)
            logging.debug(f"Updated output with section: {section_name}")

    asyncio.run(process_sections())
    logging.info("run_output_generation completed")
    return full_output

def run_async_workspace(list_of_links):
    # This function synchronously runs the asynchronous research_workspace function
    logging.info("Starting run_async_workspace")
    try:
        result = asyncio.run(research_workspace(list_of_links))
        logging.info("run_async_workspace completed successfully")
        return result
    except Exception as e:
        logging.error(f"Error in run_async_workspace: {str(e)}")
        raise
    

def run_query_generation(user_query, list_of_sections, task_context_required):
    logging.info("Starting run_query_generation")
    try:
        result = asyncio.run(query_generation(user_query, list_of_sections, task_context_required))
        logging.info("run_query_generation completed successfully")
        return result
    except Exception as e:
        logging.error(f"Error in run_query_generation: {str(e)}")
        raise

def run_information_retrieval(section_queries, query_engine):
    logging.info("Starting run_information_retrieval")
    try:
        result = asyncio.run(information_retrieval(section_queries, query_engine))
        logging.info("run_information_retrieval completed successfully")
        return result
    except Exception as e:
        logging.error(f"Error in run_information_retrieval: {str(e)}")
        raise
