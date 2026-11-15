import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent

def preprocess(base_path):
    temp1 = pd.read_csv(base_path / 'raw_data/temp_2000_to_2009.csv', encoding='utf-8-sig')
    temp2 = pd.read_csv(base_path / 'raw_data/temp_2010_to_2019.csv', encoding='utf-8-sig')
    temp3 = pd.read_csv(base_path / 'raw_data/temp_2020_to_present.csv', encoding='utf-8-sig')
    anl_co2 = pd.read_csv(base_path / 'raw_data/anl_co2.csv', encoding='utf-8-sig')
    oni = pd.read_csv(base_path / 'raw_data/oni.csv', encoding='utf-8-sig')

    temp = pd.concat([temp1, temp2, temp3])
    temp = temp.drop_duplicates()

    temp = temp.rename(columns={
    '일시': 'date',
    '평균기온(°C)': 'temp_avg',
    '최저기온(°C)': 'temp_min'
})

    anl_co2 = anl_co2.rename(columns={
    '일시': 'year',
    '평균 이산화탄소(CO2) 배경대기농도(ppm)': 'co2'
})

    oni = oni.rename(columns={
    'Year': 'year',
    'ASO': 'aso',
    'SON': 'son'
})

    temp = temp[['date', 'temp_avg', 'temp_min']]
    anl_co2 = anl_co2[['year', 'co2']]
    oni = oni[['year', 'aso', 'son']]

    temp.to_csv(base_path / 'data/temp.csv', index=False, encoding='utf-8-sig')
    anl_co2.to_csv(base_path / 'data/anl_co2.csv', index=False, encoding='utf-8-sig')
    oni.to_csv(base_path / 'data/oni.csv', index=False, encoding='utf-8-sig')

    temp, anl_co2, oni = temp.dropna(), anl_co2.dropna(), oni.dropna()

    temp['date'] = pd.to_datetime(temp['date'])

    mask = (temp['date'].dt.month == 11) & (temp['date'].dt.day.between(8, 21))
    temp = temp[mask].reset_index(drop=True)

    return temp, anl_co2, oni


if __name__ == '__main__':
    anl_co2, oni, temp = preprocess(BASE)
    print(temp.head())
    print(anl_co2.head())
    print(oni.head())
