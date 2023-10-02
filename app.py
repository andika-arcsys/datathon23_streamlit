import streamlit as st
# import streamlit.components.v1 as components
import streamlit_analytics
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Set up Streamlit
st.set_page_config(page_title="Global Temperature Forecasting Model",
                   page_icon=':chart_with_upwards_trend:', layout='wide')


# Include Google Analytics tracking code
# with open("google_analytics.html", "r") as f:
#     html_code = f.read()
#     components.html(html_code, height=0)

# Load all csv files
df = pd.read_csv("./data/Datathon_Fall2023_Dataset.csv")
df_processed = pd.read_csv("./data/new_data.csv")
df_model = pd.read_csv("./data/processed_data.csv")
df_forecasted = pd.read_csv("./data/forecast_with_y.csv")
df_refined_forecast = pd.read_csv("./data/refined_forecast_with_y.csv")
df_performance = pd.read_csv("./data/performance_table.csv")

# Set page title and description
streamlit_analytics.start_tracking()
st.title("🌡️Temperature Anomaly Forecasting")
st.markdown("Explore temperature anomaly data and related variables to create a forecasting model using [Prophet by Facebook](https://facebook.github.io/prophet/).")

# Raw and processed data comparison
st.header("Data Comparison")
raw_data_column, processed_data_column = st.columns(2)

with raw_data_column:
    st.subheader("Raw Data")
    raw_df_col, raw_description_col = st.columns(2)
    with raw_df_col:
        st.write(df)
    with raw_description_col:
        st.markdown('''
                    This is the raw data provided. It is extracted from the [GCAG website](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/data-info)
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
    
# Seasonality Chart:
def seasonality_chart():
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
    
    stl = STL(df['Anomaly'], seasonal=11)
    result = stl.fit()
    
    # Plot the original, trend, and seasonal lines
    original_trace = go.Scatter(x=df.index, y=df['Anomaly'], mode='lines', name='Original')
    trend_trace = go.Scatter(x=df.index, y=result.trend, mode='lines', name='Trend', line=dict(color='orange'))
    seasonal_trace = go.Scatter(x=df.index, y=result.seasonal, mode='lines', name='Seasonal', line=dict(color='green'))
    
    fig = go.Figure(data=[original_trace, trend_trace, seasonal_trace])
    fig.update_layout(
                      xaxis_title='Year',
                      yaxis_title='Temperature Anomaly'
                      )
    st.plotly_chart(fig, use_container_width=True)

# Jointplot which shows the seasons distrubtion vs. anomaly
def season_jointplot():
    jointplot_col, des_col = st.columns(2)
    with jointplot_col:
        st.markdown('## Jointplot: Year vs. Anomaly with Seasonal Distribution')
        jointplot = px.scatter(df_processed, x='Year', y='Anomaly', color='Season', 
                            marginal_x='histogram', marginal_y='histogram'              
                            )
        jointplot.update_layout(height=500, margin={"r": 160, "t": 50, "l": 20, "b": 50})
        st.plotly_chart(jointplot)
    with des_col:
        st.markdown('''
                The jointplot provides a comprehensive view of the relationship between years, temperature anomalies, and seasons. It's a powerful tool for understanding both the overall trends in temperature anomalies over time and the seasonal variations within the dataset.
                
                ### Seasonal Distribution Insights:
                - **Summer:** Dominant presence suggests consistent temperature patterns during summer months, deviating from the average.
                - **Fall:** Significant presence indicates notable temperature anomalies during fall, contributing substantially to the dataset.

                ### Scatter Plot Observations:
                - **Fall:** Prevalence of red points signifies specific years with significant temperature anomalies during fall.
                - **Winter:** Occasional light blue points highlight specific years with notable temperature deviations in winter.
                - **Hints of Other Colors:** Sparse appearances of dark blue (Spring) and other colors indicate varied anomalies across all seasons.
                
                ### Changepoint
                Important value where the trend steadily increased which is used the model parameters to better fit the data.

                These patterns reveal strong seasonal variations, especially in __summer__ and __fall__, with occasional anomalies in other seasons. Further analysis can provide insights into regional climate trends and the impact of specific seasons on temperature anomalies.

                ''')

# Some model code and explanation
def code_container():
    st.markdown('''
                ### Gist of the Model Fitting & Training
                View full code on [github](https://github.com/mahakanakala/datathon23/blob/main/prophet_model.ipynb) ''')
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
   
# Prediction vs. Actual Plot 
def prediction_plot():
    st.markdown('''
                ## Actual vs. Initial Prediction vs. Refined Prediction Plot
                Shows the actual, initial forecast, and refined forecast values plotted. The refined forecast values are much closer to the actual.
                ### How we refined prediction
                __Add Changepoints__: `changepoints=['1891-09-01', '1939-12-01', '1975-01-01', '2012-03-20', '2010-04-06'])`.
                Choose 4 critical areas of immediate shift in the graph to better the data to the model  
                ''')
    fig = go.Figure()

    # Plotting actual data
    fig.add_trace(go.Scatter(x=df_forecasted['ds'], y=df_forecasted['y'], mode='lines', name='Actual', line=dict(color='blue')))

    # Plotting intial predicted data
    fig.add_trace(go.Scatter(x=df_forecasted['ds'], y=df_forecasted['yhat'], mode='lines', name='Intial Prediction', line=dict(color='red')))
    
    # Adding upper and lower bounds for initial predictions
    fig.add_trace(go.Scatter(x=df_forecasted['ds'], y=df_forecasted['yhat_lower'], mode='lines', fill=None, line=dict(color='rgba(255,0,0,0.3)'), name='Intial Predicted Lower Bound'))
    fig.add_trace(go.Scatter(x=df_forecasted['ds'], y=df_forecasted['yhat_upper'], mode='lines', fill='tonexty', line=dict(color='rgba(255,0,0,0.3)'), name='Intial Predicted Upper Bound'))
    
    # Plotting intial predicted data
    fig.add_trace(go.Scatter(x=df_refined_forecast['ds'], y=df_refined_forecast['yhat'], mode='lines', name='Refined Prediction', line=dict(color='green')))

    # Adding upper and lower bounds for refined predictions
    fig.add_trace(go.Scatter(x=df_refined_forecast['ds'], y=df_refined_forecast['yhat_lower'], mode='lines', fill=None, line=dict(color='rgba(122, 211, 143, 1)'), name='Refined Predicted Lower Bound'))
    fig.add_trace(go.Scatter(x=df_refined_forecast['ds'], y=df_refined_forecast['yhat_upper'], mode='lines', fill='tonexty', line=dict(color='rgba(122, 211, 143, 1)'), name='Refined Predicted Upper Bound'))

    # Customize layout
    fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Value',
                    hovermode='x')
    st.plotly_chart(fig, use_container_width=True)

# Model Performance Statistics and Analysis
def performance_stats():
    st.subheader('Model Performance Statistics')
    stats_df_col, stats_des_col = st.columns(2)
    with stats_df_col:
        st.write(df_performance)
        st.markdown(
        '''
        ### Analysis
        - Not an accurate model despite various other important regressors (energy use, water use, greenhouse gas emmisions) that are missing in the model. 
        
        ### Further Things to Refine Model
        - **Additive or Multiplicative Seasonality**: Prophet allows you to model the seasonality as either additive or multiplicative. Depending on your data, one might be more appropriate than the other. Experiment with both to see which fits better.
        - **Adding Regressors**: As mentioned before, adding more data variables can tune the prediction better
        > By Maha Kanakala, Nikhila Sundar, Nivedha Sundar
        '''
    )
    with stats_des_col:
        st.markdown(
            '''
            **Horizon:** This represents the number of days into the future the model is predicting.

            **MSE (Mean Squared Error):**
            MSE measures the average of the squares of the errors. Lower values indicate better accuracy. It provides an overall idea of how well the model is performing.

            **RMSE (Root Mean Squared Error):**
            RMSE is the square root of the MSE. It gives you the average error in the same units as your target variable. RMSE is especially useful when large errors are particularly undesirable.

            **MAE (Mean Absolute Error):**
            MAE measures the average of the absolute errors. It's not as sensitive to outliers as MSE, making it a good metric when the data contains outliers. MAE provides a more interpretable understanding of the model's accuracy.

            **MAPE (Mean Absolute Percentage Error):**
            MAPE represents the mean percentage difference between the predicted and actual values. It's useful for understanding the scale of the errors in relation to the actual values. MAPE is expressed as a percentage, making it easy to interpret.

            **MDAPE (Median Absolute Percentage Error):**
            MDAPE is similar to MAPE, but it uses the median instead of the mean. It's less sensitive to outliers in the data, providing a more robust measure of prediction accuracy, especially when dealing with skewed data.

            **SMAPE (Symmetric Mean Absolute Percentage Error):**
            SMAPE is another percentage-based error metric that is symmetric, meaning it doesn't disproportionately weigh overestimates and underestimates. It provides a balanced view of the model's performance, considering both positive and negative errors.

            **Coverage:**
            Coverage likely refers to prediction intervals. It measures the proportion of actual values that fall within the prediction intervals. A higher coverage percentage indicates that your prediction intervals are more accurate, providing a measure of the model's reliability and confidence in its predictions.

            These metrics collectively offer a comprehensive assessment of the forecasting model's accuracy, precision, and reliability, providing valuable insights into its performance over various prediction horizons.
            '''
        )
    
# Chatbot on the sidebar contextualized to the project
# Model training
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

# Contextualization
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
    prediction_plot()
    performance_stats()
        
if __name__ == '__main__':
    main()
    
streamlit_analytics.stop_tracking()