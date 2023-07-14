import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import maya
from streamlit_echarts import st_echarts
from st_on_hover_tabs import on_hover_tabs
from streamlit_option_menu import option_menu
from icecream import ic
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import requests
from datetime import datetime
import json
import boto3
from prophet.serialize import model_to_json, model_from_json
from prophet import Prophet
import os
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from st_aggrid import AgGrid
from stqdm import stqdm
import itertools
from streamlit_autorefresh import st_autorefresh
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_folium import st_folium
import folium



stqdm.pandas()

#### FUNCTIONS ##############
#############################

def ManageDateTime(ds_obj):
    return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:%M:%S")
def ManageDate(ds_obj):
    return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d")
def ManageDateMinute(ds_obj):
    return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:%M:00")
def ManageDateHour(ds_obj):
    return datetime.strftime(maya.parse(ds_obj).datetime(), "%Y-%m-%d %H:00:00")


def Train_Prophet(train_data, weather_data,
                  metric,
                  params_grid = {  
                        'changepoint_prior_scale': [ 0.01, 0.1],
                        'weekly_seasonality': [10,15],
                        'daily_seasonality':  [10, 15],
                        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                        "seasonality_mode": ["multiplicative", "additive"]
                    }, 
                    weather_features = "all",
                  exogenous_data = pd.DataFrame()):

    """
    
    Trains a prophet model based on train

    Parameters
    ----------
    train_data: Data in the format ["ds", "y"], where ds is a columns in datetime or string = "%Y-%m-%d %H:%M:%S
    weather_data: Weather Data in the format ["ds", values]. The expected values to be included are: shortwave_radiation,temperature_2m,
                direct_radiation, diffuse_radiation, direct_normal_irradiance. If they are not in the weather data will not be added to
                the model.
    params_grid: Grid of parameters with which the model will be tested. There is a default grid defined.
    exogenous_data: Exogenous data

    Returns
    -------
    Best Model: The model 
    """
    
    # Generate all combinations of parameters
    all_params = [dict(zip(params_grid.keys(), v)) for v in itertools.product(*params_grid.values())]
    cutoffs = pd.to_datetime(['2020-09-01', '2020-10-01'])
    rmses = []  # Store the RMSEs for each params here
    mae = [] # Store the MAEs for each params here
    coverage = [] # Store the Coveragess for each params here

    train_data["ds"] = train_data["ds"].apply(ManageDateMinute)
    train_data = train_data[["ds", "y"]].groupby("ds").mean().reset_index(level = "ds")
    train_data["ds_hour"] = train_data["ds"].apply(ManageDateHour)
    weather_data["ds_hour"] = weather_data["ds_hour"].apply(str)

    train_data = pd.merge(train_data, weather_data, on = "ds_hour")
    # Incluir grid search cross fit. 
    
    if metric in ["MAE", "RMSE"]:
        best_metric = float("inf")
    elif metric == "Coverage":
        best_metric = 0
    else:
        metric = "MAE"
        best_metric = float("inf")

    best_model = None

    if exogenous_data.shape[0] != 0:
        train_data["ds_day"] = train_data["ds"].apply(ManageDate)
        exogenous_data.columns = ["ds_day", "people"]
        train_data = pd.merge(train_data, exogenous_data, on = "ds_day")


    for params in stqdm(all_params, desc = "Training Configurations of models"):
        m = Prophet(**params)
        m.add_country_holidays("IT")
        if weather_features == "all":
            m.add_regressor('shortwave_radiation')
            m.add_regressor('temperature_2m')
            m.add_regressor("direct_radiation")
            m.add_regressor("diffuse_radiation")
            m.add_regressor("direct_normal_irradiance")
        else:
            for feat in weather_features:
                m.add_regressor(feat)
        
        if exogenous_data.shape[0] != 0:
            for col in exogenous_data.columns.values:
                if col not in ["ds", "ds_day", "ds_hour"]:
                    m.add_regressor("people", mode = "multiplicative")

        m.fit(train_data)

        df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])
        mae.append(df_p["mae"].values[0])
        coverage.append(df_p["coverage"].values[0])

        if metric == "MAE":
            if mae[-1] < best_metric:
                best_metric = mae[-1]
                best_model = m
        elif metric == "RMSE":
             if rmses[-1] < best_metric:
                best_metric = rmses[-1]
                best_model = m
        elif metric == "Coverage":
            if coverage[-1] > best_metric:
                best_metric = coverage[-1]
                best_model = m
    
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    tuning_results["mae"] = mae
    tuning_results["coverage"] = coverage
    print(tuning_results)

    return best_model, best_metric, tuning_results

def Predict_Prophet():
    return None

#### APP CODE ##############
#############################

st.set_page_config(page_title="AI Platform - RENergetic", 
                   page_icon="ren_logo.png",
                   layout= "wide")

original_title = '<p style="font-family:Georgia, sans-serif; color:White; font-size: 80px;">AI Manager</p>'

if "file" not in st.session_state:
    st.session_state["file"] = "Not Imported"

if st.session_state["file"] == "Not Imported":

    file = st.file_uploader("Input Consumption",type = ["xlsx", "csv", "feather"])

    if file is not None:
        try:
            data = pd.read_csv(file)
        except:
            try:
                data = pd.read_excel(file)
            except:
                data = pd.read_feather(file)
        
        AgGrid(data, theme = "dark", height = 150,fit_columns_on_grid_load = True)
    
        col_0_0, col_0_1, col_0_2, col_0_3 = st.columns([1,1,1,2])

        with col_0_0:
            ds_var = st.selectbox("Select variable for datetime", options = data.columns.values)
        with col_0_1:
            value_var = st.selectbox("Select variable as time series value", options = data.drop([ds_var], axis = 1).columns.values)
        with col_0_2:
            city_name = st.text_input("Input name for city of asset")
        with col_0_3:
            vars_to_add = st.multiselect("Select exogenous variables to add to the analysis", options = data.drop([ds_var, value_var], axis = 1).columns)
        
        try:
            data_to_save = data[[ds_var, value_var]]
            data_to_save.columns = ["ds", "value"]
            data_to_save["ds"] = data_to_save["ds"].apply(lambda x: datetime.strftime(maya.parse(x).datetime(), "%Y-%m-%d %H:%M:%S"))
            fig = px.scatter(data_to_save, x = "ds", y = "value")
        except KeyError:
            fig = go.Figure()
        
        

        if city_name != "":
            if city_name != "None":
                url = "https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=10&language=en&format=json".format(city_name = city_name)
                geo_ = requests.get(url)
                try:
                    results_ = geo_.json()["results"][0]
                    lat = results_["latitude"]
                    lon = results_["longitude"]
                except KeyError:
                    lat = 0
                    lon = 0

            if lat != 0 and lon != 0:
                m = folium.Map(location = [lat, lon], zoom_start=14)
            else:
                m = None

            
        
        col_results_0, col_results_1 = st.columns(2)

        with col_results_0:
            st.plotly_chart(fig)
        with col_results_1:
            try:
                st_data = st_folium(m,height = 400, width = 600)
            except:
                st.info("No city input")

        if st.button("Download data"):
            data_to_save.to_feather("data_model.feather")
            st.session_state["file"] = "Imported"
            count_files = st_autorefresh(interval=10, limit=2, key="train_model")


            


else:
    col_title_0, col_title_1, col_title_2 = st.columns([1,6,4])

    with col_title_0:
        st.image("ren_logo.png")

    with col_title_1:
        st.markdown(original_title, unsafe_allow_html=True)

    st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

    selected3 = option_menu("",["Dashboard", "Merged Data",  "Forecasting", "Explainability"], 
            icons=['file-bar-graph-fill', 'bezier', "lightning-fill", "book-half"], 
            menu_icon="cast", default_index=0, orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#75a133"},
                "icon": {"color": "#000000", "font-size": "25px"}, 
                "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#232d4b"},
                "nav-link-selected": {"background-color": "#003290"},
            }
        )

    data = pd.read_feather("data_model.feather")
    data["ds"] = data["timestamp"].apply(lambda x: datetime.strftime(maya.parse(x).datetime(), "%Y-%m-%d %H:%M:%S"))
    weather_data = pd.read_feather("weather_data_milan.feather")
    weather_data = weather_data[weather_data.ds_hour <= "2023-01-01"]


    if selected3 == "Dashboard":

        fig = go.Figure()
        fig = fig.add_trace(go.Scatter(x = data.ds, y = data.MW,
                                    line = dict(color='#75a133', width=1),
                                    marker = dict(cauto = True),
                                    mode='lines'))
        fig.update_layout(
                xaxis=dict(
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor='rgb(204, 204, 204)',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Poppins',
                        size=12,
                        color='rgb(117,161,51)',
                    )
                ),
                yaxis = dict(
                    showgrid = False,
                    linewidth = 2,
                    ticks = "outside",
                    tickfont = dict(
                        family = "Poppins",
                        size = 14,
                        color = 'rgb(117,161,51)'
                    ),
                    showticklabels = True
                ),
                showlegend=False,
                title = "Asset Time Series Segreate Dibit 2",
                plot_bgcolor='rgb(15,18,22)',
                xaxis_title='',
                yaxis_title='Power (MW)',
                height = 635
            )


        col_1_0, col_1_1 = st.columns(2)

        with col_1_0:
            st.plotly_chart(fig, use_container_width= True)
        with col_1_1:
            select_feature = st.selectbox("Weather Feature", weather_data.drop("ds_hour", axis = 1).columns.values)
            fig_2 = go.Figure()
            fig_2 = fig_2.add_trace(go.Scatter(x = weather_data.ds_hour, y = weather_data[select_feature],
                                    line = dict(color='#005573', width=1),
                                    marker = dict(cauto = True),
                                    mode='lines'))
            fig_2 = fig_2.update_layout(
                xaxis=dict(
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor='rgb(204, 204, 204)',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Poppins',
                        size=12,
                        color='rgb(117,161,51)',
                    )
                ),
                yaxis = dict(
                    showgrid = False,
                    linewidth = 2,
                    ticks = "outside",
                    tickfont = dict(
                        family = "Poppins",
                        size = 14,
                        color = 'rgb(117,161,51)'
                    ),
                    showticklabels = True
                ),
                showlegend=False,
                title = "Weather Data Milano",
                plot_bgcolor='rgb(15,18,22)',
                xaxis_title='',
                yaxis_title='Power (MW)',
                height = 550
            )
            
            
            st.plotly_chart(fig_2, use_container_width=True)

    elif selected3 == "Merged Data":
        data["ds"] = data["ds"].apply(ManageDateMinute)
        train_data = data[["ds", "MW"]].groupby("ds").mean().reset_index(level = "ds")
        train_data["ds_hour"] = train_data["ds"].apply(ManageDateHour)
        weather_data["ds_hour"] = weather_data["ds_hour"].apply(str)

        train_data = pd.merge(train_data, weather_data, on = "ds_hour")

        col_select_0, col_select_1 = st.columns(2)

        with col_select_0:
            select_var_1 = st.selectbox("Variable 1", train_data.drop("ds", axis = 1).columns.values)
        with col_select_1:
            select_var_2 = st.selectbox("Variable 2", train_data.drop("ds", axis = 1).columns.values)
        
        if select_var_1 == select_var_2:
            train_data["select_2"] = train_data[select_var_1].shift(24)
            train_data.fillna(-1)
            train_data = train_data[train_data["select_2"] >= 0]
        else:
            train_data["select_2"] = train_data[select_var_2].copy()
        
        fig = go.Figure()
        fig = fig.add_trace(go.Scatter(x = train_data[select_var_1], y = train_data["select_2"],
                                    line = dict(color='#75a133', width=1),
                                    marker = dict(cauto = True),
                                    mode='markers'))
        fig.update_layout(
                xaxis=dict(
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor='rgb(204, 204, 204)',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Poppins',
                        size=12,
                        color='rgb(117,161,51)',
                    )
                ),
                yaxis = dict(
                    showgrid = False,
                    linewidth = 2,
                    ticks = "outside",
                    tickfont = dict(
                        family = "Poppins",
                        size = 14,
                        color = 'rgb(117,161,51)'
                    ),
                    showticklabels = True
                ),
                showlegend=False,
                title = "Correlate Energy and Weather Features",
                plot_bgcolor='rgb(15,18,22)',
                xaxis_title=select_var_1,
                yaxis_title= select_var_2,
                height = 635
            )

        st.plotly_chart(fig, use_container_width= True)
        
    elif selected3 == "Forecasting":

        if 'forecasted' not in st.session_state:
            st.session_state['forecasted'] = False

        if st.session_state["forecasted"] == False:
            col_date_min, col_date_max = st.columns(2)

            with col_date_min:
                date_min = st.date_input("Start Date Train", maya.parse(data["ds"].tolist()[0]).datetime(), maya.parse(data["ds"].tolist()[0]).datetime(), maya.parse(data["ds"].tolist()[-1]).datetime())

            with col_date_max:
                date_max = st.date_input("Start Date Train", maya.parse(data["ds"].tolist()[-1]).datetime(), date_min, maya.parse(data["ds"].tolist()[-1]).datetime())

            weather_features = st.multiselect(
                'Weather Features',
                weather_data.drop(["ds_hour"], axis =1).columns.values,
                weather_data.drop(["ds_hour"], axis =1).columns.values)

            weather_features_final = []
            if len(weather_features) == len(weather_data.drop(["ds_hour"], axis =1).columns.values):
                weather_features_final = "all"

            
            if st.button("Train Model"):
                data["y"] = data["MW"].copy()
                # best_model, best_metric, tuning_results = Train_Prophet(data, weather_data, "MAE")
                # tuning_results.to_feather("tuning_results.feather")
                st.session_state["forecasted"] = True
                count2 = st_autorefresh(interval=10, limit=2, key="train_model")
        
        else:
            tuning_results = pd.read_feather("tuning_results.feather")
            future = pd.read_feather("future.feather")
            forecast = pd.read_feather("forecast.feather")
            future["ds"] = future["ds"].apply(str)
            forecast["ds"] = forecast["ds"].apply(str)
            merged_df = pd.merge(future, forecast, on = "ds")
            merged_df["y_baseline"] = merged_df["y"].shift(24)
            merged_df = merged_df.fillna(np.mean(merged_df.y))
            merged_df = merged_df[merged_df.ds > "2020-09-07"]
            
            r2_baseline = r2_score(merged_df.y, merged_df.y_baseline)
            r2_model = r2_score(merged_df.y, merged_df.yhat)
            delta_r2 = np.round(((r2_model- r2_baseline)/r2_baseline)*100, 2)

            mae_baseline = mean_absolute_error(merged_df.y, merged_df.y_baseline)
            mae_model = mean_absolute_error(merged_df.y, merged_df.yhat)
            delta_mae = np.round(((mae_baseline- mae_model)/mae_baseline)*100, 2)

            rmse_baseline = np.sqrt(mean_squared_error(merged_df.y, merged_df.y_baseline))
            rmse_model = np.sqrt(mean_squared_error(merged_df.y, merged_df.yhat))
            delta_rmse = np.round(((rmse_baseline- rmse_model)/rmse_baseline)*100, 2)

            with st.expander("Tuning Results"):
                AgGrid(tuning_results, theme = "dark", height = 150,fit_columns_on_grid_load = True)

            col_metric_0, col_metric_1, col_metric_2 = st.columns(3)

            with col_metric_0:
                st.metric(label = "R2", value = "{val}".format(val = np.round(r2_model, 3)), delta=f'{delta_r2} %')
            with col_metric_1:
                st.metric(label = "MAE", value = "{val}".format(val = np.round(mae_model, 3)), delta=f'{delta_mae} %')
            with col_metric_2:
                st.metric(label = "R2", value = "{val}".format(val = np.round(rmse_model, 3)), delta=f'{delta_rmse} %')

            style_metric_cards(border_left_color = "#75a133", background_color = "#000000", border_size_px = 6, border_radius_px = 10)
            
            fig = go.Figure()
            fig = fig.add_trace(go.Scatter(x = future[future.ds > "2020-11-07"].ds, y = future[future.ds > "2020-11-07"].y, name = "real",
                                    line = dict(color='#75a133', width=1),
                                    marker = dict(cauto = True),
                                    mode='lines'))
            fig = fig.add_trace(go.Scatter(x = forecast[forecast.ds > "2020-11-07"].ds, y = forecast[forecast.ds > "2020-11-07"].yhat, name = "forecast",
                                    line = dict(color='#005573', width=2),
                                    marker = dict(cauto = True),
                                    mode='lines'))
            fig = fig.add_trace(go.Scatter(x = forecast[forecast.ds > "2020-11-07"].ds, y = forecast[forecast.ds > "2020-11-07"].yhat_lower, name = "forecast_lower",
                                    line = dict(color='#00aa9b', width=0.5, dash = "dot"),
                                    marker = dict(cauto = True),
                                    mode='lines'))
            fig = fig.add_trace(go.Scatter(x = forecast[forecast.ds > "2020-11-07"].ds, y = forecast[forecast.ds > "2020-11-07"].yhat_upper, name = "forecast_upper",
                                    line = dict(color='#fbdad9', width=0.5, dash = "dot"),
                                    marker = dict(cauto = True),
                                    mode='lines'))

            fig.update_layout(
                xaxis=dict(
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor='rgb(204, 204, 204)',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Poppins',
                        size=12,
                        color='rgb(117,161,51)',
                    )
                ),
                yaxis = dict(
                    showgrid = False,
                    linewidth = 2,
                    ticks = "outside",
                    tickfont = dict(
                        family = "Poppins",
                        size = 14,
                        color = 'rgb(117,161,51)'
                    ),
                    showticklabels = True
                ),
                showlegend=True,
                title = "Asset Time Series Segreate Dibit 2 Forecast",
                plot_bgcolor='rgb(15,18,22)',
                xaxis_title='',
                yaxis_title='Power (MW)',
                height = 635
            )
            st.plotly_chart(fig, use_container_width= True)

            if st.button("Retrain"):
                st.session_state["forecasted"] = False
                count = st_autorefresh(interval=10, limit=2, key="retrain")
            
    elif selected3 == "Explainability":

        forecast = pd.read_feather("forecast.feather")

        from plotly.subplots import make_subplots

        fig = make_subplots(rows = 4, cols = 1, 
                            subplot_titles = ("Trend", "Daily Seasonality", "Weekly Seasonality", "Temperature"))
        fig.add_trace(go.Scatter(x = forecast.ds, y = forecast.trend, name = "trend"), row = 1, col = 1)
        fig.add_trace(go.Scatter(x = forecast[forecast.ds >= "2020-11-11"].ds, y = forecast[forecast.ds >= "2020-11-11"].daily, name = "daily seasonality"), row = 2, col = 1)
        fig.add_trace(go.Scatter(x = forecast[forecast.ds >= "2020-11-04"].ds, y = forecast[forecast.ds >= "2020-11-04"].weekly, name = "weekly seasonality"), row = 3, col = 1)
        fig.add_trace(go.Scatter(x = forecast.ds, y = forecast.temperature_2m, name = "Temperature"), row = 4, col = 1)
        fig.update_layout(showlegend=False, title_text="Components of Model",height=1000, width=600)

        texts_ = ["Value", "Percentage", "Percentage", "Percentage"]

        for i in range(1,5):
            fig.update_xaxes(showline = True, showgrid = False, ticks = "outside", tickfont = dict(family='Poppins',size=12,color='rgb(117,161,51)'), row = i, col = 1)
            fig.update_yaxes(showline = True, showgrid = False, title_text = texts_[i-1],ticks = "outside", tickfont = dict(family='Poppins',size=12,color='rgb(117,161,51)'), row = i, col = 1)


        st.plotly_chart(fig, use_container_width= True)
