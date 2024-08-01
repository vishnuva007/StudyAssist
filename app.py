import streamlit as st
from llm_functions import load_data, split_text, initialize_llm, generate_questions, create_retrieval_qa_chain
import os
# Hardcode your Azure OpenAI credentials here
openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')  
openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')  
print(openai_endpoint)
# Initialization of session states
if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['questions_list'] = []
    st.session_state['submitted'] = False

with st.container():
    st.markdown("# Study Assist")

# Let user upload a file
uploaded_file = st.file_uploader("Upload your study material", type=['pdf'])

if uploaded_file is not None:
    # Load data from PDF
    text_from_pdf = load_data(uploaded_file)

    # Split text for question generation
    documents_for_question_gen = split_text(text_from_pdf, chunk_size=10000, chunk_overlap=200)

    # Split text for question answering
    documents_for_question_answering = split_text(text_from_pdf, chunk_size=500, chunk_overlap=200)

    # Initialize large language model for question generation
    llm_question_gen = initialize_llm(openai_api_key=openai_api_key, model="gpt-4-32k", temperature=0.4)

    # Initialize large language model for question answering
    llm_question_answering = initialize_llm(openai_api_key=openai_api_key, model="gpt-4-32k", temperature=0.1)

    # Create questions if they have not yet been generated
    if st.session_state['questions'] == 'empty':
        with st.spinner("Generating questions..."):
            st.session_state['questions'] = generate_questions(llm=llm_question_gen, chain_type="refine", documents=documents_for_question_gen)

    if st.session_state['questions'] != 'empty':
        # Show questions on screen
        st.info(st.session_state['questions'])

        # Split questions into a list
        st.session_state['questions_list'] = st.session_state['questions'].split('\n')

        with st.form(key='my_form'):
            st.session_state['questions_to_answers'] = st.multiselect(label="Select questions to answer", options=st.session_state['questions_list'])
            submitted = st.form_submit_button('Generate answers')
            if submitted:
                st.session_state['submitted'] = True

        if st.session_state['submitted']:
            # Initialize the Retrieval QA Chain
            with st.spinner("Generating answers..."):
                answer_chain = create_retrieval_qa_chain(openai_api_key=openai_api_key, documents=documents_for_question_answering)
                # For each question, generate an answer
                for question in st.session_state['questions_to_answers']:
                    # Generate answer
                    answer = answer_chain(question)
                    # Show answer on screen
                    st.write(f"Question: {question}")
                    st.info(f"Answer: {answer}")
