import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

st.set_page_config(page_title="Global Temperature Forecasting Model",
                   page_icon=':thought_balloon:', layout='wide')

# Load the raw and processed csv data files from github
df = pd.read_csv("./data/Datathon_Fall2023_Dataset.csv")
df_processed = pd.read_csv("./data/new_data.csv")
df_model = pd.read_csv("./data/processed_data.csv")

# Set page title and description
st.title("🌡️Temperature Anomaly Forecasting")
st.markdown("Explore temperature anomaly data and related variables to create a forecasting model using [Prophet by Facebook](https://facebook.github.io/prophet/).")

st.header("Data Comparison")
raw_data_column, processed_data_column = st.columns(2)

with raw_data_column:
    st.subheader("Raw Data")
    df_col, description_col = st.columns(2)
    with df_col:
        st.write(df)
    with description_col:
        st.markdown('''
                    This is the raw data provided. It is extracted from the [GCAG website](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/data-info)" . 
                    ### Columns
                    - Year: year + month (Example: 18501 = 1850, January)
                    - Anomaly: Percent change from last month to current month
                    ''')

with processed_data_column:
    st.subheader("Processed Data")
    df_col, description_col = st.columns(2)
    with df_col:
        st.write(df_processed)
    with description_col:
        st.markdown('''
                    We split the year column into year and month. We added a column for season to test our hypothesis that seasons have an impact on the temperature anomaly.
                    ### Columns
                    - Year: Year
                    - Month: Month
                    - Anamoly: Anomaly percetnage from last month to current month
                    - Season: New column to check the seasonality of the data set, seasons grouped like this:
                        - (Winter: 1-3, Spring: 4-6, Summer: 7-9, Fall: 10-12)
                    ''')
    
# Seasonality:
def seasonality_chart():
    seasonal_period = 11
    st.markdown('''
                ## Seasonality and Trends in Temperature Anomalies
                *Visualize raw data and check the trends and seasonality.*  
                **Trends**: represents the long-term movement or direction in the data. It captures the underlying pattern in the data that shows a 
                general increase, decrease, or stability over time.  
                **Seasonality**: refers to the repeating pattern or fluctuations in the data that occur at regular intervals, typically within a year. 
                These patterns repeat over a fixed period, such as monthly, quarterly, or yearly, and are often influenced by seasonal factors like 
                weather, holidays, or biological cycles. In this case the seasonality is yearly.''')
    # Ensure 'Anomaly' column has numerical values (convert to float if necessary)
    df['Anomaly'] = pd.to_numeric(df['Anomaly'], errors='coerce')  # Drop rows with NaN values in 'Anomaly' column
    
    # Convert 'Year' to datetime format and set it as the index
    df['Year'] = pd.to_datetime(df['Year'], format='%Y%m')
    df.set_index('Year', inplace=True)
    
    stl = STL(df['Anomaly'], seasonal=seasonal_period)  # Seasonal period is set to 12 for monthly data
    result = stl.fit()
    
    original_trace = go.Scatter(x=df.index, y=df['Anomaly'], mode='lines', name='Original')
    trend_trace = go.Scatter(x=df.index, y=result.trend, mode='lines', name='Trend', line=dict(color='orange'))
    seasonal_trace = go.Scatter(x=df.index, y=result.seasonal, mode='lines', name='Seasonal', line=dict(color='green'))
    
    fig = go.Figure(data=[original_trace, trend_trace, seasonal_trace])
    fig.update_layout(
                      xaxis_title='Year',
                      yaxis_title='Temperature Anomaly',
                    #   margin={"r": 0, "t": 0, "l": 0, "b": 0}, 
                    #   height=800, 
                    #   width=1050
                      )
    st.plotly_chart(fig, use_container_width=True)

 # Jointplot
jointplot_col, des_col = st.columns(2)

with jointplot_col:
    def season_jointplot():
        st.markdown('## Jointplot: Year vs. Anomaly with Seasonal Distribution')
        jointplot = px.scatter(df_processed, x='Year', y='Anomaly', color='Season', 
                            marginal_x='histogram', marginal_y='histogram' 
                             
                            )
        st.plotly_chart(jointplot)

with des_col:
    st.markdown('''
                This jointplot shows the correlation between the seasons and the anamoly.
                ''')

# Model Code + explanation
def code_container():
    st.markdown('''
                ### Gist of the Model Fitting & Training
                View full code on [github]() ''')
    st.code('''
            # Import libraries
            from prophet import Prophet
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            %matplotlib inline
            
            # Split test & Train Data (75% & 25%)
            train = df[:1554]
            test = df[1554:]
            
            # Initialize and fit model
            model = Prophet().fit(train)
            future = model.make_future_dataframe(periods=518, freq='MS')
            forecast = model.predict(future)
            ''', language='python')

# Chatbot model training
model_name = "deepset/roberta-base-squad2"

# Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.sidebar.header("Question Answering Chatbot")
st.sidebar.markdown('''
                    Made using [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2?context=You+are+a+LLM+built+to+knowledgable+on+my+data+science+project+submission.+This+is+a+project+about+the+forecasting+model+of+global+temperatures.+The+data+set+is+from+GCAG.+You+are+hosted+on+a+Streamlit+app.&question=What+is+the+data?). 
                    Ask any questions about our project!
                    ''')
# User input box
question = st.sidebar.text_input("Enter your question:")

context = '''
            You are an LLM built to be knowledgeable on a data science project submission to the Rutgers Data Science Fall 23 Datathon 
            submitted by Maha, Nikhila, and Nivedha. This is a project about the forecasting. The data set is from GCAG, this is the website 
            that has more info on the data: https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/data-info. 
            You are hosted on a Streamlit app. This project is about a forecasting model of global temperatures built using Prophet by Facebook. 
            The raw data has Year and Anomaly Column. The anamoly representsanomaly means a departure from a reference value or long-term average. 
            A positive anomaly indicates that the observed temperature was warmer than the reference value, while a negative anomaly indicates 
            that the observed temperature was cooler than the reference value. The visualizations are built plotly - an interactive python 
            plotting library. The github link to the streamlit is: https://github.com/mahakanakala/datathon23_streamlit.
            The github link to the model and training of this project is: https://github.com/mahakanakala/datathon23.
            The processed data has Year, Anomaly, Month, and Season Column.
'''

# Intializing deepset/roberta-base-squad2 model and getting answer instances from pipeline
if question:
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {
        'question': question,
        'context': context
    }
    answer = nlp(QA_input)
    st.sidebar.write("Answer:", answer['answer'])
    
def main():
    seasonality_chart()
    season_jointplot()
    code_container()
        
if __name__ == '__main__':
    main()