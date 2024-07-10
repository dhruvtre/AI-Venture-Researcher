### implemented 10 batch size for firecrawl
### error handling if search results return an error
### updated output format to markdown. 

import io
import os
from openai import OpenAI
from openai import AsyncOpenAI
import openai
from pprint import pprint
import time
from timeit import default_timer as timer
from langchain_community.document_loaders import FireCrawlLoader
from tqdm import tqdm
import logging
import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.write("OpenAI_API_Key", st.secrets["openai_api_key"])
st.write("Firecrawl_API_Key", st.secrets["firecrawl_api_key"])
st.write("Search_API_Key", st.secrets["search_api_key"])

client = OpenAI(api_key=st.secrets["openai_api_key"]) ### ENTER YOUR FIRECRAWL API KEY HERE
start_time = timer()


def openai_call_wo_tools(model, message_history):
    response = client.chat.completions.create(
    model=model,
    messages=message_history,
    temperature = 0.8
    )
    response_message = response.choices[0].message
    response_message_content = response_message.content
    return response_message_content

def openai_w_tools(messages, tools, tool_choice, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

#task classification system

def task_classification(user_query):
  task_classification_task_prompt = f'''You are a text classification AI agent. Please review the following user query and classify it in one of the four listed task categories.

  ###USER QUERY
  {user_query}

  ###TASK LABELS
  create_macro_industry_reports
  create_company_profiles

  Your final output should only include one most relevant label and it should be in the following format:
  ###TASK LABEL
  [label determined]

  For example:
  AGENT: ###create_macro_industry_reports
  AGENT: ###create_company_profiles'''
  task_classification_message_history = [{"role": "user",
                                          "content": task_classification_task_prompt}]
  task_classifciation_response = openai_call_wo_tools("gpt-3.5-turbo", task_classification_message_history)
  logging.info(f"Task label generated: {task_classifciation_response}")

  macro_reports_format = '''### Macro Industry Report

#### 1. Industry Overview
- **Industry name/sector**: 
- **Brief description**: (2-3 sentences)
- **Key sub-sectors or segments**: 
- **Major players**: 
- **Recent developments**
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#): 

#### 2. Market Size and Growth
- **Total Addressable Market (TAM)**: 
- **Current market size**: 
- **Historical growth rate**: 
- **Projected growth rate** (next 3-5 years): 
- **Key growth drivers**
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#): 

#### 3. Competitive Landscape
- **Major competitors**: 
- **Market share distribution**: 
- **Recent M&A activity**: 
- **Barriers to entry**: 
- **Competitive strategies**
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#): 

#### 4. Technology and Trends
- **Current technological disruptions**: 
- **Emerging technologies**: 
- **Adoption rates of new technologies**: 
- **Key innovations**: 
- **Impact on industry**
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#): 

#### 5. Investment and Future Outlook
- **Recent notable investments**: 
- **Active VCs and strategic investors**: 
- **Emerging investment themes**: 
- **Short-term industry forecast** (1-2 years): 
- **Long-term industry projections** (5-10 years):
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#)'''

  company_profile_format = '''### Company Profile

#### 1. Company Basics
- **Company name**: 
- **Industry/sector**: 
- **Founded date**: 
- **Brief description** (1-2 sentences): 
- **Headquarters location**: 
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#)

#### 2. Product/Service
- **Core offering description**: 
- **Target market/customer segments**: 
- **Key pain points addressed**: 
- **Unique selling proposition**: 
- **Key differentiators from competitors**: 
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#)

#### 3. Financials
- **Total funding raised**: 
- **Last funding round details** (if available): 
- **Revenue model** (how they make money): 
- **Current revenue** (if available): 
- **Profitability status**: 
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#)

#### 4. Team
- **Founder(s) name(s) and brief background**: 
- **Key team members**: 
- **Advisors and board members**: 
- **Notable hires**: 
- **Team size and growth rate**: 
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#)

#### 5. Traction and Market
- **Key metrics** (e.g., number of users, customers, growth rate)**: 
- **Notable partnerships or clients** (if any)**: 
- **Market size (TAM)**: 
- **Key competitors**: 
- **Recent market trends**: 
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#)

#### 6. Recent Developments
- **Latest news or milestones**: 
- **Upcoming product launches or expansions**: 
- **Strategic initiatives**: 
- **Industry awards or recognitions**: 
- **Future plans and roadmap**: 
- **Citations/References**: 
  - [Reference 1](#)
  - [Reference 2](#)'''

  if task_classifciation_response == "###create_company_profiles":
    return task_classifciation_response, company_profile_format
  elif task_classifciation_response == "###create_macro_industry_reports":
    return task_classifciation_response, macro_reports_format

def existing_task_context(user_query, task_classification_label, output_format):
  macro_reports_context = '''What is the specific industry or sector, including sub-sectors or verticals of interest?
  What is the geographic focus of the industry research task? Is it global, regional, or country-specific analysis?
  What is the time frame for the research, including historical data coverage and future projection or forecast period?'''

  company_profile_context = '''What is the specific industry or sector, including sub-sectors or verticals of interest?
  What is the geographic focus of the industry research task? Is it global, regional, or country-specific analysis?
  What is the time frame for the research, including historical data coverage and future projection or forecast period?'''
  if task_classification_label == "###create_company_profiles":
    task_context_required = company_profile_context
  elif task_classification_label == "###create_macro_industry_reports":
    task_context_required = macro_reports_context
  existing_context_chat = [{"role": "user",
                            "content": f'''You are an experienced VC analyst reviewing a task request for a {task_classification_label} from a senior human Partner. 
                            
                            Please review the below user_query and the list of context required as well as the desired output format, and summarise any existing context about the task already present. Try your best to infer context from the user query.
                            ###USER_QUERY
                            {user_query}

                            ###TASK CONTEXT REQUIRED
                            {task_context_required}

                            ####OUTPUT FORMAT
                            {output_format}

                            Your output should only include any existing context, under the relevant context question.'''}]

  existing_context = openai_call_wo_tools("gpt-4-turbo", existing_context_chat)
  print("Existing context gathered")
  return existing_context, task_context_required

def task_context_collection(existing_context, task_classification_label, user_query, task_context_required):
    task_context_collection_task_prompt = f'''You are an experienced VC Analyst doing {task_classification_label} for a user. Here is the complete text of the user query:

  ###USER QUERY
  {user_query}

  Your main goal in the current conversation is to ask the user three short questions to capture all the context required.

  To do {task_classification_label}, you must know the following:

  ###TASK CONTEXT REQUIRED
  {task_context_required}

  Here is what you already know.

  ###EXISTING CONTEXT
  {existing_context}

  Ask questions to the user. Do not ask them for context that you already have.'''
    questions_for_task_context_chat = [{"role": "user",
                                        "content": task_context_collection_task_prompt}]
    questions_for_task_context = openai_call_wo_tools("gpt-4-turbo", questions_for_task_context_chat)
    return questions_for_task_context

def task_context_summarisation(existing_context, task_classification_label, user_query, task_context_required, questions_for_task_context, additional_answers):
    task_context_summarisation_task_prompt = f'''You are an experienced VC Analyst doing {task_classification_label} for a user. Here is the complete text of the user query:

  ###USER QUERY
  {user_query}

  Your main goal in the current conversation is to summarise the task context required to complete this task. 

  To do {task_classification_label}, you must know the following:

  ###TASK CONTEXT REQUIRED
  {task_context_required}

  Here is what you already know.

  ###EXISTING CONTEXT - Gathered from the original task message.
  {existing_context}

  ###USER CLARIFICATIONS TO QUESTIONS
  ###QUESTIONS
  {questions_for_task_context}
  ###CLARIFICATIONS
  {additional_answers}

  Summarise the task context in one concise format. This will be used as context shared across an AI workflow to research and complete the task. Here is a recommended output format: 
  
  ####RECOMMENDED OUTPUT FORMAT
  # TASK QUERY - {user_query}
  # SUMMARY OF TASK CONTEXT: 
  #     GEOGRAPHICAL FOCUS:
  #     DURATION: 
  #     ADDITIONAL NOTES:
  
  Only output a TASK CONTEXT passage in a fixed format.'''

    task_context_summary_conversation = [{"role": "user", 
                                          "content": task_context_summarisation_task_prompt}]
    
    summarised_task_context = openai_call_wo_tools("gpt-4-turbo", task_context_summary_conversation)
    return summarised_task_context


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

import requests
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

def task_planning(task_context_message, task_classification_label, user_query, tool_set, output_format, existing_research=""):

  task_planning_system_prompt = f'''You are an experienced VC Analyst helping your manager, a principal partner at a VC fund at doing tasks.
  To do this, here is all the information that will be shared with you:
  1. Task_classification_label: This is a label for the specific task you are currently working on.
  2. User_query: The task request statement shared by your manager.
  3. Task_context_message: Additional context on the task collected.
  4. Existing_research (optional): A list of research already done if available.

  Your main goal is to create a step by step plan of gathering all relevant information to complete this task. To do this you have access to the following tools:

  ###TOOLS AVAILABLE
  {tool_set}

  In the case Existing research is provided, your goal is to build on top of it.

  Your final output should always be in the format specified below.

  Here are some examples:
  USER: "Dealey, please provide a list of the most recent funding deals in the healthcare AI space, including the startup names, funding amounts, and lead investors. This will help us identify potential investment opportunities."
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

  Your final output should only be a list of tool calls in the format above. Only output in the specific format.’’’'''

  task_planning_task_prompt = f'''
  ###TASK CLASSIFICATION LABEL
  {task_classification_label}

  ###USER QUERY
  {user_query}

  ###ALL TASK CONTEXT
  {task_context_message}

  ####EXPECTED OUTPUT FORMAT
  {output_format}

  ####EXISTING RESEARCH
  {existing_research}'''

  task_planning_chat_history = [{"role": "system",
                                 "content": task_planning_system_prompt},
                                {"role": "user",
                                 "content": task_planning_task_prompt}]
  task_planning = openai_call_wo_tools("gpt-4-turbo", task_planning_chat_history)
  return task_planning

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
        for tool_call in tool_calls:
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

FirecrawlAPI_key = f'Bearer {st.secrets["firecrawl_api_key"]}'  # Replace with your actual Firecrawl API key

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
                                headers={'Authorization': FirecrawlAPI_key},
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
    return list_of_links

from llama_index.core import VectorStoreIndex, Document, get_response_synthesizer
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore

openai.api_key = st.secrets["openai_api_key"]
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.embed_model = OpenAIEmbedding()

def chunking_research_ouptut(list_of_links):
    documents = []
    chunk_size = 1000
    overlap = 200
    for entry in list_of_links: 
        for item in entry:
            page_content = item.get("page_text", "")
            if page_content:
                chunks_page_content = [page_content[i:i+chunk_size] for i in range(0, len(page_content), chunk_size-overlap)]
                for chunk in chunks_page_content:
                    chunk_metadata = {"link_url": item.get("link_url", "url_na"),
                                      "link_title": item.get("link_title", "title_na")}
                    doc = Document(text=chunk, metadata=chunk_metadata)
                    documents.append(doc)
            else:
                continue
    print("Documents created")
    print(len(documents))
    return documents

def query_engine_generation(documents):
    index = VectorStoreIndex.from_documents(documents)
    retriever_1 = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        vector_store_info='''Index of documents containing "text" which are chunks from webpages and "metadata" which includes the link_url and link_title for the webpage that the chunk is from.'''
    )
    # assemble query engine
    query_engine_1 = RetrieverQueryEngine(
        retriever=retriever_1
    )
    print("Query engine generated.")
    return query_engine_1

def query_engine_running(query_engine, user_query):
    response_2 = query_engine.query(user_query)
    response_format = {"response": response_2.response, 
                       "response_sources": response_2.source_nodes[0].node.get_metadata_str()}
    return response_format

def query_generation(user_query, task_context_required, output_format):
  #generating list of queries

  query_generation_system_prompt = f'''You are an experienced VC Analyst creating the final output of the following task for a user, a principal partner at a VC fund. Here is all the information available to you:
  ###USER QUERY
  {user_query}

  ###ALL TASK CONTEXT
  {task_context_required}

  ####OUTPUT FORMAT
  {output_format}

  Here is the format that this company report should follow: 

  Your current task is to generate a list of queries to search for this information from a research workspace which includes a variety of resources for the task.

  Your final output should include nothing but the list of queries in the specified format below.
  
  ####OUTPUT FORMAT
  
  <QUERIES>
  <QUERY>"enter query text"</QUERY>
  <QUERY>"enter query 2 text"</QUERY>'''

  query_generation_chat_history = [{"role": "system",
                                 "content": query_generation_system_prompt}]
  query_generation = openai_call_wo_tools("gpt-4-turbo", query_generation_chat_history)
  return query_generation

import re

def information_retrieval(query_gen_text, query_engine):
    list_of_answers = []
    queries = re.findall(r'<QUERY>\s*"([^"]+)"\s*</QUERY>', query_gen_text)
    for query in queries: 
        query_response = query_engine_running(query_engine, query)
        answer_set = {"query": query,
                 "answer": query_response}
        list_of_answers.append(answer_set)
    # Save list_of_answers to a JSON file
    with open('list_of_answers.json', 'w') as json_file:
        json.dump(list_of_answers, json_file, indent=4)
    return list_of_answers

def output_generation(user_query, task_context_required, list_of_answers, output_format):
  output_generation_system_prompt = f'''You are an experienced VC Analyst creating the final output of the following task for a user, a principal partner at a VC fund. Here is all the information available to you:

  ###USER QUERY
  {user_query}

  ###ALL TASK CONTEXT
  {task_context_required}

  ###TASK SPECIFIC RESEARCH
  {list_of_answers}

  ####OUTPUT FORMAT
  {output_format}

  Your main goal is to output the information available to you in a fixed one-pager format.

  The document should include all relevant citations.

  Your final output should include nothing but the output in the specified format.'''

  output_generation_chat_history = [{"role": "system",
                                 "content": output_generation_system_prompt}]
  output_generation = openai_call_wo_tools("gpt-4-turbo", output_generation_chat_history)
  return output_generation

def run_async_workspace(list_of_links):
    # This function synchronously runs the asynchronous research_workspace function
    return asyncio.run(research_workspace(list_of_links))

  
import streamlit as st

# Assuming these functions are defined elsewhere in your codebase
# from your_module import task_classification, existing_task_context, task_context_collection, task_context_summarisation, task_planning, plan_executor, run_async_workspace, chunking_research_ouptut, query_engine_generation, query_generation, information_retrieval, output_generation

# Initialize session state variables
if 'label' not in st.session_state:
    st.session_state.label = ""
if 'task_context_r' not in st.session_state:
    st.session_state.task_context_r = ""
if 'output_format_r' not in st.session_state:
    st.session_state.output_format_r = ""
if 'task_submitted' not in st.session_state:
    st.session_state.task_submitted = False
if 'additional_context_submitted' not in st.session_state:
    st.session_state.additional_context_submitted = False

st.title("Dealey - Version 1")
st.write("You can use me to create company profiles or industry reports.")
st.write("I currently only have access to Google Search but I will get more tools super soon.")

# Task submission form
with st.form(key='task_form'):
    task_statement = st.text_input('Please describe the task I can help you with today.')
    submit_button = st.form_submit_button('Submit')

    if submit_button and task_statement:
        st.session_state.task_submitted = True

# Process task if submitted
if st.session_state.task_submitted:
    task_label_run, output_format_return = task_classification(task_statement)
    st.session_state.label = task_label_run
    st.markdown(f"Task identified as ***'{st.session_state.label}'***")
    st.session_state.output_format_r = output_format_return
    st.write('Understanding your query!')
    
    existing_context_gathered, task_context_required_run = existing_task_context(task_statement, st.session_state.label, st.session_state.output_format_r)
    st.session_state.task_context_r = existing_context_gathered
    st.write('Checking for clarifications.')
    
    message = task_context_collection(existing_context_gathered, st.session_state.label, task_statement, task_context_required_run)
    with st.chat_message("Dealey"):
        st.write(message)
    
    # Additional context form
    with st.form(key='additional_context_form'):
        additional_context = st.text_input('Please provide additional context.')
        share_button = st.form_submit_button('Share Additional Context')
        
        if share_button and additional_context:
            st.session_state.additional_context_submitted = True

    # Process additional context if submitted
    if st.session_state.additional_context_submitted:
        final_context_gathered = task_context_summarisation(existing_context_gathered, st.session_state.label, task_statement, task_context_required_run, message, additional_context)
        st.session_state.task_context_r = final_context_gathered
        st.markdown("Thank you for sharing necessary details. I am ready to get started!")
        
        with st.status("Doing the work...", expanded=True):
            st.write("Generating research plan now.")
            task_plan_run = task_planning(final_context_gathered, st.session_state.label, task_statement, tool_set, st.session_state.output_format_r, existing_research="")
            st.write('Research plan generated.')
            st.write('Executing research plan now.')
            plan_results = plan_executor(task_plan_run, tool_set)
            st.write('Research completed.')
            st.write('Reading research material now.')
            research_results = run_async_workspace(plan_results)
            st.write('Reading complete. Indexing material now!')
            research_run_docs = chunking_research_ouptut(research_results)
            query_engine_run = query_engine_generation(research_run_docs)
            queries_run = query_generation(task_statement, st.session_state.task_context_r, st.session_state.output_format_r)
            st.write('Indexing complete. Finding answers now.')
            list_of_answers_run = information_retrieval(queries_run, query_engine_run)
            st.write('Starting to work on output now.')
            final_output = output_generation(task_statement, st.session_state.task_context_r, list_of_answers_run, st.session_state.output_format_r)
            end_time = timer()
            logging.info(f"Total_time_taken: {end_time - start_time}")
            st.markdown(final_output)

elif not task_statement:
    st.error("Please enter a task description.")
    
    
