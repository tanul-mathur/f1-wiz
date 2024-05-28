####### IMPORT LIBRARIES
from dotenv import load_dotenv
import os
import streamlit as st
import datetime
import fastf1 as ff1
# import fastf1.plotting

# langchain libraries
from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

# import butler.get_agent as get_agent
from utils.butler import get_agent

######## INITIAL SETUP

# Enable the cache
ff1.Cache.enable_cache('cache') 
# ff1.plotting.setup_mpl(misc_mpl_mods=False)

st.set_page_config(page_title='🏁 F1 Wiz 🧙',
                #    page_icon="🏁",
                   layout = 'wide')
st.title('🏎️💨💨  :red[F1 Wiz] 🧙')
st.write('### Ask questions about any Formula 1 session !!')
# left_co, cent_co,last_co = st.columns(3)
# with cent_co:
st.image("https://media1.tenor.com/m/RtrDuGASCoMAAAAd/f1.gif", use_column_width = True)

# https://media.giphy.com/media/1QjxtwZ9LoPD2Jlcuq/giphy.gif
# https://media.giphy.com/media/cC9Ue0I59m5NEJTlMH/giphy.gif
# https://media.giphy.com/media/MovqJSMROh1gA/giphy.gif

###### INITIATE SIDEBAR AND ASK FOR OPENAI API KEY 
with st.sidebar:
    # openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/api-keys)"
    "[View the source code](https://github.com/tanul-mathur/f1-wiz/blob/main/f1_results_agent.py)"

###### SELECTION OPTIONS 
with st.container():
    st.write('#### Enter your OpenAI API Key :')
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.write('#### Select Session details :')
    col1, col2, col3 = st.columns(3)

    year = col1.selectbox('Year',list(range(2019, datetime.date.today().year+1)))
    event_details_df = ff1.get_event_schedule(year)
    event_name = col2.selectbox('Event',
                event_details_df.loc[event_details_df['EventDate'].dt.date < datetime.date.today(), 
                                    'EventName'])

    selected_event = event_details_df.loc[event_details_df['EventName'] == event_name,]
    event_round = selected_event['RoundNumber'].values[0] 
    event_loc = selected_event['Location'].values[0] 
    event_date = selected_event['EventDate'].values[0]

    session_name = col3.selectbox('Session', ['Practice 1', 'Practice 2', 'Practice 3', 'Sprint', 'Sprint Shootout', 'Qualifying', 'Race'])

    go_button = st.button('Go!')
    
    if 'go_clicked' not in st.session_state.keys():
            st.session_state['go_clicked'] = False

# on click
if go_button | st.session_state['go_clicked']:
    
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
        st.stop()
    if openai_api_key.startswith('sk-'):

        st.session_state['go_clicked'] = True

        ####### LOAD THE APP
        session = ff1.get_session(year, event_name, session_name)
        session.load()

        st.write(f"### You selected the :red[{event_name}] :red[{year}], Session : {session_name}")

        results_df = session.results
        filtered_df = results_df.loc[:,['TeamName','FullName','CountryCode','Position','ClassifiedPosition','GridPosition','Q1','Q2','Q3','Time','Status','Points']]
        # all_laps = session.laps

        st.write('The table provides driver and result information for all drivers that participated in a session.')
        with st.expander("See summary of the referenced table"):
            st.write("## Results table")
            st.write(filtered_df.head())

        # INITIATE AGENT
        pandas_df_agent = get_agent(openai_api_key, df = results_df)

        # Initialize the chat messages history
        if "messages" not in st.session_state.keys(): 
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! How can I help?"}
            ]
        # Display the prior chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # User input area
        prompt = st.chat_input("Your question")
        # if st.button("Send"): 
        if prompt: # Ensure the prompt is not empty
            st.session_state.messages = [{"role": "user", "content": prompt}]
            # Save user input to chat history
            # st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display the updated chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # If the last message is from the user, generate a new response from the assistant
            if st.session_state.messages[-1]["role"] == "user":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
                        st.write(response)
                        # Save assistant response to chat history
                        message = {"role": "assistant", "content": response}
                            # st.session_state.messages.append({"role": "assistant", "content": response})

