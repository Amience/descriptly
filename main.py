import streamlit as st
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
import os
import openai
import hmac

# Function to create the database
def create_database(df):
    database = []

    for index, row in df.iterrows():
        entry = {}
        entry['title'] = row['Title']

        characteristics_list = []

        for col in df.columns:
            if col.startswith('Characteristics:'):
                characteristic_name = col.split(':')[1]
                characteristic_value = row[col]
                if pd.notna(characteristic_value) and characteristic_value != 'none':
                    characteristics_list.append(f"{characteristic_name}: {characteristic_value}")

        entry['characteristics'] = '\n'.join(characteristics_list)
        database.append(entry)

    return database


def check_password():
    """Checks and validates the user's password, managing login state."""

    def password_entered(username, password):
        """Validates credentials and updates session state accordingly."""
        if username in st.secrets["passwords"] and hmac.compare_digest(password, st.secrets["passwords"][username]):
            st.session_state["password_correct"] = True
            st.session_state["username"] = username  # Store username in session state
            st.experimental_rerun()  # Rerun the script to update the UI and hide the form
        else:
            st.session_state["password_correct"] = False
            st.error("User not known or password incorrect")

    if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
        with st.form("login_form"):
            username = st.text_input("Username", key="username_input")
            password = st.text_input("Password", type="password", key="password_input")
            submitted = st.form_submit_button("Log in")
            if submitted:
                password_entered(username, password)


# Call the check_password function to manage login state
check_password()

# Stop the application if not authenticated
if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
    st.stop()

# Display a welcome message and other content if logged in
if "username" in st.session_state:
    st.write(f"Welcome back, {st.session_state['username']}!")



with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        key="chatbot_api_key",
        type="password",
    )


if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()
else:
    openai_api_key = openai_api_key
    st.session_state['OPENAI_API_KEY'] = openai_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    openai.api_key = st.session_state['OPENAI_API_KEY']

    # Create a column layout
    col1, col2 = st.columns([1, 5])
    # Display the logo in the first column
    with col1:
        st.image('images/logo.webp', use_column_width=True)
    # Display the startup name in the second column
    with col2:
        st.markdown("<h1 style='text-align: left;'>Descriptly</h1>", unsafe_allow_html=True)
        st.markdown(
            '''<p style="font-family:sans-serif; color:Green; font-size: 16px;"> \
            Automate and Elevate Your Product Descriptions
            </p>''',
            unsafe_allow_html=True)

    uploaded_file = st.file_uploader('Upload a CSV file')
    # initialise df as pandas data frame and number of duplicates
    df_original = pd.DataFrame()

    if uploaded_file is not None:
        df_original = pd.read_csv(uploaded_file)
        if 'original_df' not in st.session_state:
            st.session_state.original_df = df_original
        # Create a copy of original_df to save AI results
        if 'modified_original_df' not in st.session_state:
            st.session_state.modified_original_df = st.session_state.original_df.copy()
        if 'ai_results' not in st.session_state:
            # Initialize list to store AI results
            st.session_state.ai_results = []

        st.title("Original dataset")
        st.write(st.session_state.original_df)

        with st.sidebar:
            st.title("Generate a dataset for AI")
            if st.button('Generate'):
                database = create_database(st.session_state.original_df)
                # Convert database to DataFrame
                st.session_state.database_df = pd.DataFrame(database)
            st.divider()

        # uncomment for debugging
        #if 'database_df' in st.session_state and not st.session_state.database_df.empty:
        #    st.title('Dataset for AI')
        #    st.write(st.session_state.database_df)
        #    st.divider()

        if 'database_df' in st.session_state and not st.session_state.database_df.empty:
            with st.sidebar:
                st.title('Configure Your AI')
                # Dropdown menu for language selection
                language = st.selectbox(
                    'Select a language:',
                    ('English', 'Spanish', 'German', 'Russian', 'Ukrainian')
                )
                document = st.multiselect(
                    "Select document",
                    options=['HTML formatted', 'Amazon', 'e-Bay'],
                    default=['HTML formatted']  # Set default selections to match the unique_subparts
                )
                filter_selection = st.selectbox('Filter:',
                                                ('Row 1', 'Row 1-5','All')
                                                )

                if st.button('Generate AI'):
                    # Resetting ai_results and making a fresh copy of original_df for processing
                    st.session_state.ai_results = []
                    st.session_state.modified_original_df = st.session_state.original_df.copy()

                    # Depending on filter selection, choose rows to process
                    if filter_selection == 'Row 1-5':
                        data_to_process = st.session_state.database_df.head(5)
                    if filter_selection == 'Row 1':
                        data_to_process = st.session_state.database_df.head(1)
                    if filter_selection == 'All':
                        data_to_process = st.session_state.database_df

                    total_tokens = 0
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_cost = 0
                    number_of_rows = len(data_to_process)
                    st.write(f"""processing {number_of_rows} rows of data""")

                    # Setup the progress bar
                    progress_bar = st.progress(0)
                    total_rows = len(data_to_process)
                    processed_rows = 0

                    # Process each row
                    for idx, row in data_to_process.iterrows():
                        title = row['title']
                        specification = row['characteristics']

                        if 'HTML formatted' in document:
                            prompt_text = f"""You are a marketing expert and a very proficient SEO writer for a product listing on Amazon. \
        Your task is to write SEO-optimised product description in {language} language following the guidelines below. 
        
        GUIDELINES:
        1. Write an engaging product title that accurately describes the product's functionality and clearly highlights \
        the benefits of the product to potential customer. Make sure the title is between 150 and 200 characters. 
        2. Write an SEO optimised product description section without a headline. Provide a comprehensive overview of the product's features \
        and benefits to help customers make informed purchase decisions. Make sure this section is at least 300 words long. 
        3. Write an SEO optimised 'Key Features' section. This section should be formatted into bullet points with \
        a maximum of 600 characters for each bullet point. Each bullet point should sufficiently describe the product's features \
        and benefits and should include the information about product specification. Ensure each bullet point begins \
        with 2 or 3 words in capital letters indicating its category.
        4. Write a clear and concise conclusion paragraph without a headline. 
        
        WRITING STYLE:
        1. Write in {language} language. 
        2. Be friendly, engaging and clear and concise, but also professional
        3. Write in a conversational style in active voice
        4. Format all your responses and headlines and title with HTML tags (e.g. <br>, <strong></strong>, <ul></ul><li></li> etc.) ready for webpage integrations. \
        5. Make sure the title and all headings are in bold
        6. Don't write 'title' ahead of the actual title
        7. Dont write 'Product Description' and 'Conclusions' heading ahead of each of those sections
        
        THE PRODUCT:
        Title: {title}
        Specification:
        {specification}
        """
                        prompt = ChatPromptTemplate.from_template(prompt_text)
                        #llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
                        #llm = ChatOpenAI(model_name="gpt-4-0125-preview")
                        llm = ChatOpenAI(model_name="gpt-4o")
                        chain = prompt | llm | StrOutputParser()

                        with get_openai_callback() as cb:
                            result = chain.invoke({"title": title, "specification": specification, "language": language})
                            st.session_state.ai_results.append(result)
                            # Sum up the tokens and cost for each row
                            total_tokens += cb.total_tokens
                            prompt_tokens += cb.prompt_tokens
                            completion_tokens += cb.completion_tokens
                            total_cost += cb.total_cost

                        # Update the progress bar
                        processed_rows += 1
                        progress_bar.progress(processed_rows / total_rows)

                    # Reset the progress bar for next use
                    progress_bar.empty()
                    # Save AI results to the copy of the DataFrame
                    st.session_state.modified_original_df.loc[data_to_process.index, 'AI_result'] = st.session_state.ai_results

                    with st.sidebar:
                        # Display summary of total tokens and costs after all rows are processed
                        st.write(f"Total Tokens: {total_tokens}")
                        st.write(f"Prompt Tokens: {prompt_tokens}")
                        st.write(f"Completion Tokens: {completion_tokens}")
                        st.write(f"Total Cost (USD): ${total_cost:.2f}")
                        st.divider()



        if st.session_state.ai_results:
            st.title("Modified dataset with AI column added at the end")
            st.dataframe(st.session_state.modified_original_df)

        if 'ai_results' in st.session_state and st.session_state.ai_results:
            st.write("Audit the results:")

            # Input for selecting the row number
            row_number = st.number_input('Enter the row number you want to display', min_value=1,
                                         max_value=len(st.session_state.ai_results), step=1)

            if row_number:
                row_index = row_number - 1  # Convert to zero-based index
                if row_index < len(st.session_state.ai_results):
                    # Check if the row index is within the bounds of the database_df
                    if row_index < len(st.session_state.database_df):
                        title = st.session_state.database_df.iloc[row_index]['title']
                        specifications = st.session_state.database_df.iloc[row_index]['characteristics']

                        # Convert specification to a list style display
                        spec_lines = specifications.split('\n')  # Split the specifications into lines
                        formatted_specs = '\n'.join(
                            f'- {line}' for line in spec_lines)  # Prepend with dash for markdown list formatting

                        # Displaying input data without showing index
                        input_data = pd.DataFrame({
                            'Title': [title],
                            'Specification': [formatted_specs]  # Display formatted specifications
                        })
                        st.write("Input data for the selected row:")
                        st.write(input_data.set_index(input_data.columns[0]))


                    st.markdown(f"**Row {row_number} Output (RAW):**")
                    st.write(st.session_state.ai_results[row_index])
                    st.divider()
                    st.markdown(f"**Row {row_number} Output (formatted):**")
                    st.markdown(st.session_state.ai_results[row_index], unsafe_allow_html=True)
                else:
                    st.error("The selected row number is out of range.")



