import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent

# 실제 수능 시행일 (시행 연도: 날짜)
SUNEUNG_DATES = {
    2000: '2000-11-15', 2001: '2001-11-07', 2002: '2002-11-06', 2003: '2003-11-05',
    2004: '2004-11-17', 2005: '2005-11-23', 2006: '2006-11-16', 2007: '2007-11-15',
    2008: '2008-11-13', 2009: '2009-11-12', 2010: '2010-11-18', 2011: '2011-11-10',
    2012: '2012-11-08', 2013: '2013-11-07', 2014: '2014-11-13', 2015: '2015-11-12',
    2016: '2016-11-17', 2017: '2017-11-23', 2018: '2018-11-15', 2019: '2019-11-14',
    2020: '2020-12-03', 2021: '2021-11-18', 2022: '2022-11-17', 2023: '2023-11-16',
    2024: '2024-11-14', 2025: '2025-11-13',
}

def preprocess(base_path):
    temp1 = pd.read_csv(base_path / 'raw_data/temp_2000_to_2009.csv', encoding='utf-8-sig')
    temp2 = pd.read_csv(base_path / 'raw_data/temp_2010_to_2019.csv', encoding='utf-8-sig')
    temp3 = pd.read_csv(base_path / 'raw_data/temp_2020_to_present.csv', encoding='utf-8-sig')
    co2 = pd.read_csv(base_path / 'raw_data/co2.csv', encoding='utf-8-sig')
    oni = pd.read_csv(base_path / 'raw_data/oni.csv', encoding='utf-8-sig')

    temp = pd.concat([temp1, temp2, temp3])
    temp = temp.drop_duplicates()

    temp = temp.rename(columns={
    '일시': 'date',
    '평균기온(°C)': 'temp_avg',
    '최저기온(°C)': 'temp_min'
    })

    co2 = co2.rename(columns={
    '일시': 'year',
    '평균 이산화탄소(CO2) 배경대기농도(ppm)': 'co2'
    })

    oni = oni.rename(columns={
    'Year': 'year',
    'ASO': 'aso',
    'SON': 'son'
    })

    temp = temp[['date', 'temp_avg', 'temp_min']]
    co2 = co2[['year', 'co2']]
    oni = oni[['year', 'aso', 'son']]

    temp, co2, oni = temp.dropna(), co2.dropna(), oni.dropna()

    temp['date'] = pd.to_datetime(temp['date'])
    mask = (temp['date'].dt.month == 11) & (temp['date'].dt.day.between(8, 21))
    temp = temp[mask].reset_index(drop=True)

    oni = oni[oni['year'] >= 2000].reset_index(drop=True)

    temp.to_csv(base_path / 'data/temp.csv', index=False, encoding='utf-8-sig')
    co2.to_csv(base_path / 'data/co2.csv', index=False, encoding='utf-8-sig')
    oni.to_csv(base_path / 'data/oni.csv', index=False, encoding='utf-8-sig')

    return temp, co2, oni


def make_suneung_temp(base_path):
    """원데이터에서 실제 수능날 서울 기온만 골라 data/suneung_temp.csv로 저장"""
    files = ['temp_2000_to_2009.csv', 'temp_2010_to_2019.csv', 'temp_2020_to_present.csv']
    suneung_temp = pd.concat(
        [pd.read_csv(base_path / 'raw_data' / f, encoding='utf-8-sig') for f in files]
    ).drop_duplicates()

    suneung_temp = suneung_temp.rename(columns={
        '일시': 'date',
        '평균기온(°C)': 'temp_avg',
        '최저기온(°C)': 'temp_min'
    })
    suneung_temp = suneung_temp[['date', 'temp_avg', 'temp_min']]
    suneung_temp['date'] = pd.to_datetime(suneung_temp['date'])

    dates = pd.to_datetime(list(SUNEUNG_DATES.values()))
    suneung_temp = suneung_temp[suneung_temp['date'].isin(dates)].reset_index(drop=True)
    suneung_temp['year'] = suneung_temp['date'].dt.year

    suneung_temp.to_csv(base_path / 'data/suneung_temp.csv', index=False, encoding='utf-8-sig')
    return suneung_temp


if __name__ == '__main__':
    temp, co2, oni = preprocess(BASE)
    suneung_temp = make_suneung_temp(BASE)
    print(temp.head())
    print(co2.head())
    print(oni.head())
    print(suneung_temp.head())
