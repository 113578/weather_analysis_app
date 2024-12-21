import multiprocessing
import requests
import json
import aiohttp
import asyncio
import pandas as pd

from sklearn.linear_model import LinearRegression


SEASONS = {'Зима': 'winter', 'Весна': 'spring', 'Осень': 'autumn', 'Лето': 'summer'}


def get_weather_data(city: str, df: pd.DataFrame) -> pd.DataFrame:
    # Фильтруем город.
    df = df[df['city'] == city].copy()

    # Расчёт минимальной, максимальной и средней температуры.
    min_temperature = df['temperature'].min()
    max_temperature = df['temperature'].max()
    mean_temperature = df['temperature'].mean()

    # Рассчитываем аномальные значения.
    anomalies = df.copy()
    
    anomalies['moving_average'] = anomalies['temperature'].rolling(window=30, min_periods=1).mean()
    anomalies['moving_std'] = anomalies['temperature'].rolling(window=30, min_periods=1).std()
    anomalies['is_anomaly'] = anomalies.apply(
        lambda column: 
            1 if 
                (column['temperature'] >= column['moving_average'] + 2 * column['moving_std']) |\
                (column['temperature'] <= column['moving_average'] - 2 * column['moving_std'])
            else 0,
        axis=1
    )
    anomalies = anomalies[['timestamp', 'temperature', 'is_anomaly']]

    # Получаем профиль сезона.
    season_profile = df.copy()
    season_profile = df.groupby('season')['temperature'].agg(average='mean', std='std')
    
    # Вычисляем тренд.
    trend = df.copy()
    
    trend['timestamp_ordinal'] = pd.to_datetime(trend['timestamp'])
    trend['timestamp_ordinal'] = trend['timestamp_ordinal'].map(pd.Timestamp.toordinal)

    X = trend[['timestamp_ordinal']]
    y = trend[['temperature']]
    
    regressor = LinearRegression()
    regressor.fit(X=X, y=y)
    trend['trend'] = regressor.predict(X=X)
    
    trend = trend[['timestamp', 'trend']]

    return {
        city: [
            mean_temperature, 
            min_temperature, 
            max_temperature,
            season_profile,
            trend,
            anomalies
        ]
    }

def collect_weather_data_multiprocess(cities: str, df: pd.DataFrame):
    temp_weather_data = []
    weather_data = {}

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        args = [(city, df) for city in cities]
        temp_weather_data = pool.starmap(get_weather_data, args)
    
    for temp_weather_object in temp_weather_data:
        weather_data.update(temp_weather_object)

    return weather_data

async def async_get_temperatures(city: str, api_key: str) -> float:
    base_url = 'https://api.openweathermap.org/data/2.5/weather?'
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url=base_url, params=params) as response:
            temperature = await response.text()
            temperature = json.loads(temperature)['main']['temp']

    return {city: temperature}

async def collect_temperatures(cities: list, api_key: str) -> dict:
    temperature = {}
    tasks = [async_get_temperatures(city=city, api_key=api_key) for city in cities]
    ts = await asyncio.gather(*tasks)
    
    for t in ts:
        temperature.update(t)

    return temperature

async def async_validate_temperature(cities: list, df: pd.DataFrame, api_key: str, season: dict='winter') -> str:
    validations = {}
    
    weather_data = collect_weather_data_multiprocess(cities=cities, df=df) 
    temperatures = await collect_temperatures(cities=cities, api_key=api_key)
    
    for city in cities:
        season_profile = weather_data[city][3]
        
        season_average = season_profile.loc[season, 'average']
        season_std = season_profile.loc[season, 'std']
        season_top = season_average + season_std
        season_bottom = season_average - season_std
        
        if temperatures[city] > season_top: validations.update({city: 'Слишком жарко!'})
        elif temperatures[city] < season_bottom: validations.update({city: 'Слишком холодно!'})
        else: validations.update({city: 'В норме!'})

    return validations
    