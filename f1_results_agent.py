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

######## INITIAL SETUP

# Enable the cache
ff1.Cache.enable_cache('cache') 
# ff1.plotting.setup_mpl(misc_mpl_mods=False)

st.set_page_config(page_title='🏁 F1 Wiz 🧙',
                #    page_icon="🏁",
                   layout = 'wide')
st.title('🏎️💨💨  :red[F1 Wiz] 🧙')
st.write('### Ask questions about any Formula 1 session !!')
st.image("https://media.giphy.com/media/MovqJSMROh1gA/giphy.gif", use_column_width = True)

# https://media.giphy.com/media/1QjxtwZ9LoPD2Jlcuq/giphy.gif
# https://media.giphy.com/media/cC9Ue0I59m5NEJTlMH/giphy.gif

###### SELECTION OPTIONS 
with st.container():
    st.write('### Select Session details :')
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

# go_button = st.button('Lights Out and Away we Go!!')
# on click
# if go_button:

####### LOAD THE APP
session = ff1.get_session(year, event_name, session_name)
session.load()

st.write(f"### You selected the :red[{event_name}] :red[{year}], Session : {session_name}")

results_df = session.results
# all_laps = session.laps

st.write('The table provides driver and result information for all drivers that participated in a session.')
with st.expander("See summary of the referenced table"):
    st.write("## Results table")
    st.write(results_df.head())

data_schema = '''
The 'results_df' dataframe provides driver and result information for all drivers that participated in a session. By default, the session results are indexed by driver number and sorted by finishing position.

Column descriptions : 

DriverNumber | str | The number associated with this driver in this session (usually the drivers permanent number)

BroadcastName | str | First letter of the drivers first name plus the drivers full last name in all capital letters. (e.g. 'P GASLY')

FullName | str | The drivers full name (e.g. “Pierre Gasly”)

Abbreviation | str | The drivers three letter abbreviation (e.g. “GAS”)

DriverId | str | driverId that is used by the Ergast API

TeamName | str | The team name (short version without title sponsors)

TeamColor | str | The color commonly associated with this team (hex value)

TeamId | str | constructorId that is used by the Ergast API

FirstName | str | The drivers first name

LastName | str | The drivers last name

HeadshotUrl | str | The URL to the driver's headshot

CountryCode | str | The driver's country code (e.g. “FRA”)

Position | float | The drivers finishing position (values only given if session is 'Race', 'Qualifying', 'Sprint Shootout', 'Sprint', or 'Sprint Qualifying').

ClassifiedPosition | str | The official classification result for each driver. This is either an integer value if the driver is officially classified or one of “R” (retired), “D” (disqualified), “E” (excluded), “W” (withdrawn), “F” (failed to qualify) or “N” (not classified).

GridPosition | float | The drivers starting position (values only given if session is 'Race', 'Sprint', or 'Sprint Qualifying')

Time | pd.Timedelta | The drivers total race time (values only given if session is 'Race', 'Sprint', or 'Sprint Qualifying' and the driver was not more than one lap behind the leader)

Status | str | A status message to indicate if and how the driver finished the race or to indicate the cause of a DNF. Possible values include but are not limited to ‘Finished’, ‘+ 1 Lap’, ‘Crash’, ‘Gearbox’, … (values only given if session is ‘Race’, ‘Sprint’, or ‘Sprint Qualifying’)

Points | float | The number of points received by each driver for their finishing result.
'''


prefix_prompt = f'''You are working with pandas dataframes in Python that contain Formula One race results and lap details. The dataframe is called 'df'.
{data_schema} 
You are an expert in Formula One and know all about the sport.

You should use the tools below to answer the question posed of you:

python_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer. Install all required libraries.
'''

load_dotenv()
# openai_api_key = os.environ["OPENAI_API_KEY"]
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
llm = ChatOpenAI(
    temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
)

pandas_df_agent = create_pandas_dataframe_agent(
    llm,
    df = results_df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True
    ,prefix = prefix_prompt
)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Formula One!"}
    ]

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages = [{"role": "user", "content": prompt}]

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.write(response)
            message = {"role": "assistant", "content": response}
