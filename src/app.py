import asyncio
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
from weather import (
    SEASONS,
    collect_weather_data_multiprocess, 
    async_get_temperatures,
    collect_temperatures,
    async_validate_temperature
)


is_uploaded = False
is_api_key_set = False


st.title('Погода ☀️')

st.header('Загрузка данных 🔻')

uploaded_file = st.file_uploader('Загрузите файл с данными о погоде')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    cities = df['city'].unique()
    is_uploaded = True

if is_uploaded:
    st.header('Конфигурация 🔧')
    
    season_ru = st.selectbox('Выберите сезон', ['Зима', 'Весна', 'Осень', 'Лето'])
    season = SEASONS[season_ru]
    city = st.selectbox('Выберите город', cities)

    try:
        api_key = st.text_input('Введите API-ключ для OpenWeatherMap')
        temperatures = asyncio.run(async_get_temperatures(city=city, api_key=api_key))
        is_api_key_set = True
    except:
        st.error({'cod': 401, 'message': 'Invalid API key. Please see https://openweathermap.org/faq#error401 for more info.'})
        
if is_api_key_set:
    st.header('Погодный анализ 📊')
    
    weather_data = collect_weather_data_multiprocess(cities=cities, df=df)
    
    st.write(f'**Выбранный город:** {city}')
    
    st.write(
        f"""
        **Температура**
        - Средняя: {weather_data[city][0]:.2f}°C
        - Минимальная: {weather_data[city][1]:.2f}°C
        - Максимальная: {weather_data[city][2]:.2f}°C
        """
    )
    st.write('**Профиль сезона**')
    st.dataframe(weather_data[city][3])
    
    st.write('**Погодный тренд**')
    st.line_chart(data=weather_data[city][4], x='timestamp', y='trend')
    
    st.write('**Аномальные температуры**')
    fig, ax = plt.subplots()
    sns.scatterplot(data=weather_data[city][5], x='timestamp', y='temperature', hue='is_anomaly', ax=ax)
    st.pyplot(fig)

    st.subheader(f'Нормальная ли сейчас погода для {city}, в сезон {season_ru}? 🌡️')

    if st.button('Хочу узнать'):
        temperatures = asyncio.run(collect_temperatures(cities=cities, api_key=api_key))
        
        st.write(f'Текущая погода: {temperatures[city]:.2f}°C')
    
        validations = asyncio.run(async_validate_temperature(cities=cities, df=df, api_key=api_key, season=season))
        validation = validations[city]
    
        if validation == 'В норме!':
            st.success(f'Анализ нормальности погоды в {city}: {validation}')
        else:
            st.error(f'Анализ нормальности погоды в {city}: {validation}')
