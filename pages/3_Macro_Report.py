#MACRO REPORT.PY / Built on top of Dealey5.py
from Dealey_Shared_Functions import *
from section_dictionary import REPORT_SECTIONS

TASK_LABEL = "###create_macro_industry_reports"
INDUSTRY_REPORT_SECTIONS = REPORT_SECTIONS['macro_industry_report']
OUTPUT_FORMAT = '''

#### Industry Overview
- **Industry name/sector**: 
- **Brief description**: (2-3 sentences)
- **Key sub-sectors or segments**: 
- **Major players**: 
- **Recent developments**
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name 

#### Market Size and Growth
- **Total Addressable Market (TAM)**: 
- **Current market size**: 
- **Historical growth rate**: 
- **Projected growth rate** (next 3-5 years): 
- **Key growth drivers**
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name

#### Competitive Landscape
- **Major competitors**: 
- **Market share distribution**: 
- **Recent M&A activity**: 
- **Barriers to entry**: 
- **Competitive strategies**
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name

#### Technology and Trends
- **Current technological disruptions**: 
- **Emerging technologies**: 
- **Adoption rates of new technologies**: 
- **Key innovations**: 
- **Impact on industry**
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name

#### Investment and Future Outlook
- **Recent notable investments**: 
- **Active VCs and strategic investors**: 
- **Emerging investment themes**: 
- **Short-term industry forecast** (1-2 years): 
- **Long-term industry projections** (5-10 years):
- **Citations/References**: 
  - [Reference 1](#) - Source Name
  - [Reference 2](#) - Source Name'''

macro_reports_context = '''What is the specific industry or sector, including sub-sectors or verticals of interest?
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

st.set_page_config(page_title="Macro Industry Report - Dealey", page_icon="📈")
with st.sidebar:
    st.image("Group 5-2.png", width=100)

if 'industry_name' not in st.session_state:
    st.session_state['industry_name'] = ''
if 'list_of_sections' not in st.session_state:
    st.session_state['macro_list_of_sections'] = INDUSTRY_REPORT_SECTIONS
if 'macro_final_output' not in st.session_state:
    st.session_state['macro_final_output'] = ''

st.image("Group 5-2.png", width=100)

st.write(
    """
    ### Create Macro Industry Report

    Enter an industry name to generate a comprehensive report.
    """
)

with st.form(key='industry_form'):
    industry_name = st.text_input('Enter the industry name:')
    submit_button = st.form_submit_button('Generate Report')
    st.session_state['industry_name'] = industry_name

if submit_button and industry_name:
    logging.info(f"Starting macro industry report generation for: {industry_name}")
    with st.status("Generating industry macro report..."):
        step_placeholders = [st.empty() for _ in steps]

        for i, step in enumerate(steps):
            step_placeholders[i].markdown(f"⏳ {step}")
            
        ## Generating Research Plan
        step_placeholders[0].markdown("🔄 **Generating research plan**")
        # After calling run_task_planning
        st.write(f"Task plan generated for: {st.session_state.industry_name}")
        task_plan_run = run_task_planning(task_classification_label=TASK_LABEL, user_query=st.session_state.industry_name, tool_set=tool_set, output_format=OUTPUT_FORMAT)
        logging.info(f"Research plan generated for industry: {industry_name}")
        step_placeholders[0].markdown("✅ **Generating research plan**")
            
        ## Executing Research Plan
        step_placeholders[1].markdown("🔄 **Executing research plan**")
        plan_results = plan_executor(task_plan_run, tool_set)
        logging.info(f"Research plan execution completed for industry: {industry_name}")
        step_placeholders[1].markdown("✅ **Executing research plan**")

        ## Reading Research Material
        step_placeholders[2].markdown("🔄 **Reading research material**")
        research_results = run_async_workspace(plan_results)
        logging.info(f"Research material processed for industry: {industry_name}")
        logging.info(f"Number of research results : {len(research_results)}")
        step_placeholders[2].markdown("✅ **Reading research material**")
            
        ## Indexing Materal
        step_placeholders[3].markdown("🔄 **Indexing material**")
        research_run_docs = chunking_research_ouptut(research_results)
        logging.info(f"Number of documents indexed : {len(research_run_docs)}")
        query_engine_run = query_engine_generation(research_run_docs)
        logging.info(f"Material indexed and query engine generated for industry: {industry_name}")
        queries_run = run_query_generation(user_query=st.session_state.industry_name, task_context_required='', list_of_sections=st.session_state['macro_list_of_sections'])
        logging.info(f"Number of queries generated : {len(queries_run)}")
        step_placeholders[3].markdown("✅ **Indexing material**")

        ## Querying Research Workspace
        step_placeholders[4].markdown("🔄 **Finding answers**")
        list_of_answers_run = run_information_retrieval(queries_run, query_engine_run)
        logging.info(f"Information retrieval completed for industry: {industry_name}")
        logging.info(f"Number of answers retrieved : {len(list_of_answers_run)}")
        step_placeholders[4].markdown("✅ **Finding answers**")

        ## Generating Output
        step_placeholders[5].markdown("🔄 **Working on output**")
        final_output = run_output_generation(user_query=st.session_state.industry_name, task_context_required="", section_results=list_of_answers_run, output_format=OUTPUT_FORMAT)
        logging.info(f"Output generation completed for industry: {industry_name}")
        step_placeholders[5].markdown("✅ **Working on output**")
        end_time = timer()
        logging.info(f"Total_time_taken: {end_time - start_time}")
        logging.info (f"Final Output Generated: {final_output}")
        st.session_state['macro_final_output'] = final_output

elif submit_button and not industry_name:
    st.error("Please enter an industry to research.")
    logging.error("No user input.")

