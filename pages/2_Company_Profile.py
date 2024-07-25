from Dealey_Shared_Functions import *
from section_dictionary import REPORT_SECTIONS

TASK_LABEL = "###create_company_profiles"
TASK_SECTIONS = REPORT_SECTIONS['company_profile']
OUTPUT_FORMAT = '''

#### Company Basics
- **Company name**: 
- **Industry/sector**: 
- **Founded date**: 
- **Brief description** (1-2 sentences): 
- **Headquarters location**: 
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name

#### Product/Service
- **Core offering description**: 
- **Target market/customer segments**: 
- **Key pain points addressed**: 
- **Unique selling proposition**: 
- **Key differentiators from competitors**: 
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name

#### Financials
- **Total funding raised**: 
- **Last funding round details** (if available): 
- **Revenue model** (how they make money): 
- **Current revenue** (if available): 
- **Profitability status**: 
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name

#### Team
- **Founder(s) name(s) and brief background**: 
- **Key team members**: 
- **Advisors and board members**: 
- **Notable hires**: 
- **Team size and growth rate**: 
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name

#### Traction and Market
- **Key metrics** (e.g., number of users, customers, growth rate)**: 
- **Notable partnerships or clients** (if any)**: 
- **Market size (TAM)**: 
- **Key competitors**: 
- **Recent market trends**: 
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name

#### Recent Developments
- **Latest news or milestones**: 
- **Upcoming product launches or expansions**: 
- **Strategic initiatives**: 
- **Industry awards or recognitions**: 
- **Future plans and roadmap**: 
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name'''

company_profile_context = '''What is the specific industry or sector, including sub-sectors or verticals of interest?
  What is the geographic focus of the industry research task? Is it global, regional, or country-specific analysis?
  What is the time frame for the research, including historical data coverage and future projection or forecast period?'''

steps = [
    "Generating research plan",
    "Executing research plan",
    "Reading research material",
    "Indexing material",
    "Finding answers",
    "Working on output"
]

with st.sidebar:
    st.image("/Users/dhruvtrehan/Documents/localapps/Dealey/Group 5-2.png", width=100)

if 'company_name' not in st.session_state:
    st.session_state['company_name'] = ''
if 'list_of_sections' not in st.session_state:
    st.session_state['list_of_sections'] = TASK_SECTIONS
if 'company_final' not in st.session_state:
    st.session_state['company_final'] = ''

st.image("/Users/dhruvtrehan/Documents/localapps/Dealey/Group 5-2.png", width=100)
st.write(
    """
    ### Create Company Profile

    Enter a company name to generate a comprehensive profile.
    """
)

with st.form(key='company_form'):
    company_name = st.text_input('Enter the company name:')
    submit_button = st.form_submit_button('Generate Profile')
    st.session_state.company_name = company_name

if submit_button and company_name:
    logging.info(f"Starting company profile generation for: {company_name}")

    # Before calling run_task_planning
    st.write(f"Company name before task planning: {st.session_state.company_name}")

    with st.status("Generating company profile..."):
        step_placeholders = [st.empty() for _ in steps]

        for i, step in enumerate(steps):
            step_placeholders[i].markdown(f"⏳ {step}")
            
        ## Generating Research Plan
        step_placeholders[0].markdown("🔄 **Generating research plan**")
        # After calling run_task_planning
        task_plan_run = run_task_planning(task_classification_label=TASK_LABEL, user_query=st.session_state.company_name, tool_set=tool_set, output_format=OUTPUT_FORMAT)
        logging.info(f"Research plan generated for: {company_name}")
        step_placeholders[0].markdown("✅ **Generating research plan**")
            
        ## Executing Research Plan
        step_placeholders[1].markdown("🔄 **Executing research plan**")
        plan_results = plan_executor(task_plan_run, tool_set)
        logging.info(f"Research plan execution completed for: {company_name}")
        step_placeholders[1].markdown("✅ **Executing research plan**")

        ## Reading Research Material
        step_placeholders[2].markdown("🔄 **Reading research material**")
        research_results = run_async_workspace(plan_results)
        logging.info(f"Research material processed for: {company_name}")
        logging.info(f"Number of research results: {len(research_results)}")
        step_placeholders[2].markdown("✅ **Reading research material**")
            
        ## Indexing Materal
        step_placeholders[3].markdown("🔄 **Indexing material**")
        research_run_docs = chunking_research_ouptut(research_results)
        query_engine_run = query_engine_generation(research_run_docs)
        logging.info(f"Material indexed and query engine generated for: {company_name}")
        logging.info(f"Number of documents indexed: {len(research_run_docs)}")
        queries_run = run_query_generation(user_query=st.session_state.company_name, task_context_required='', list_of_sections=st.session_state['list_of_sections'])
        logging.info(f"Number of queries generated: {len(queries_run)}")
        step_placeholders[3].markdown("✅ **Indexing material**")

        ## Querying Research Workspace
        step_placeholders[4].markdown("🔄 **Finding answers**")
        list_of_answers_run = run_information_retrieval(queries_run, query_engine_run)
        logging.info(f"Information retrieval completed for: {company_name}")
        logging.info(f"Number of answers retrieved: {len(list_of_answers_run)}")
        step_placeholders[4].markdown("✅ **Finding answers**")

        ## Generating Output
        step_placeholders[5].markdown("🔄 **Working on output**")
        final_output = run_output_generation(user_query=st.session_state.company_name, task_context_required="", section_results=list_of_answers_run, output_format=OUTPUT_FORMAT)
        logging.info(f"Output generation completed for: {company_name}")
        step_placeholders[5].markdown("✅ **Working on output**")
        end_time = timer()
        logging.info(f"Total_time_taken: {end_time - start_time}")
        logging.info (f"Final Output Generated: {final_output}")
        st.session_state['company_final'] = final_output

elif submit_button and not company_name:
    st.error("Please enter a company name.")
