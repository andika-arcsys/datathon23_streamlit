import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from hugchat import hugchat
from hugchat.login import Login

st.set_page_config(page_title="Global Temperature Forecasting Model",
                   page_icon=':thought_balloon:', layout='wide')

# Load the raw and processed csv data files from github
df = pd.read_csv("Datathon_Fall2023_Dataset.csv")
df_processed = pd.read_csv("new_data.csv")
df_model = pd.read_csv("processed_data.csv")

# Set page title and description
st.title("🌡️Temperature Anomaly Forecasting")
st.markdown("Explore temperature anomaly data and related variables to create a forecasting model using [Neural Prophet](https://neuralprophet.com/).")

st.header("Data Comparison")
raw_data_column, processed_data_column = st.columns(2)

# Display raw data in the first column
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

# Display processed data in the second column
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
    
    
jointplot_col, des_col = st.columns(2)
# Jointplot
with jointplot_col:
    def season_jointplot():
        st.markdown('## Jointplot: Year vs. Anomaly with Seasonal Distribution')
        jointplot = px.scatter(df_processed, x='Year', y='Anomaly', color='Season', 
                            marginal_x='histogram', marginal_y='histogram', 
                            height=800, 
                            width=1050)
        st.plotly_chart(jointplot)

with des_col:
    st.markdown('''
                This jointplot shows the correlation between the seasons and the anamoly.
                ''')

# Line Chart of raw data
def raw_data_line_chart():
    st.subheader("Temperature Anomaly Over Years (Line Chart)")
    fig = px.line(df_model, x='ds', y='y', title='Temperature Anomaly Over Years (1850-2022)')
    fig.update_layout(title_text="title", 
                      margin={"r": 0, "t": 0, "l": 0, "b": 0}, 
                      height=800, 
                      width=1050,
                      xaxis_title="Years",
                      yaxis_title="Anamoly",)
    st.plotly_chart(fig, use_container_width=True)
    st.write()

# Seasonality:
# def seasonality_chart():
#     seasonal_period = 12
#     stl = STL(df['Anomaly'], seasonal=seasonal_period)  # Seasonal period is set to 12 for monthly data
#     result = stl.fit()
#     original_trace = go.Scatter(x=df['Year'], y=df['Anomaly'], mode='lines', name='Original')
#     trend_trace = go.Scatter(x=df['Year'], y=result.trend, mode='lines', name='Trend', line=dict(color='orange'))
#     seasonal_trace = go.Scatter(x=df['Year'], y=result.seasonal, mode='lines', name='Seasonal', line=dict(color='green'))
    
#     fig = go.Figure(data=[original_trace, trend_trace, seasonal_trace])
#     fig.update_layout(title='Seasonality and Trends in Temperature Anomalies',
#                       xaxis_title='Year',
#                       yaxis_title='Temperature Anomaly')

def seasonality_chart():
    seasonal_period = 11
    
    # Convert 'Year' to datetime format and set it as the index
    df['Year'] = pd.to_datetime(df['Year'], format='%Y%m')
    df.set_index('Year', inplace=True)
    
    # Ensure 'Anomaly' column has numerical values (convert to float if necessary)
    df['Anomaly'] = df['Anomaly'].astype(float)  # Ensure 'Anomaly' is in numeric format
    
    stl = STL(df['Anomaly'], seasonal=seasonal_period)  # Seasonal period is set to 12 for monthly data
    result = stl.fit()
    
    original_trace = go.Scatter(x=df.index, y=df['Anomaly'], mode='lines', name='Original')
    trend_trace = go.Scatter(x=df.index, y=result.trend, mode='lines', name='Trend', line=dict(color='orange'))
    seasonal_trace = go.Scatter(x=df.index, y=result.seasonal, mode='lines', name='Seasonal', line=dict(color='green'))
    
    fig = go.Figure(data=[original_trace, trend_trace, seasonal_trace])
    fig.update_layout(title='Seasonality and Trends in Temperature Anomalies',
                      xaxis_title='Year',
                      yaxis_title='Temperature Anomaly')
    return fig

# Centering the plot
left, middle, right = st.columns((0.1, 6, 0.1))
with middle:
    raw_data_line_chart()
    seasonality_chart()
    season_jointplot()

# Model COde + explanation

# Interactive Sidebar
st.sidebar.header("ChatBot")



def main():
    season_jointplot()