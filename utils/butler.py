from dotenv import load_dotenv
import os
import streamlit as st
import datetime
# import fastf1 as ff1
# import fastf1.plotting

# langchain libraries
from langchain.agents import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

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
def get_agent(openai_api_key, df, model = "gpt-3.5-turbo", prefix_prompt = prefix_prompt):
    '''
    [ARGS]
    open_api_key    : OpenAI API key
    df              : dataframe to refrence
    model           : Model name
    '''

    
    load_dotenv()
    # openai_api_key = os.environ["OPENAI_API_KEY"]

    llm = ChatOpenAI(
        temperature=0, model=model, openai_api_key=openai_api_key, streaming=True
    )

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df = df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True
        ,prefix = prefix_prompt
    )

    return pandas_df_agent
