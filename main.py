import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import preprocess

BASE = preprocess.BASE


def main():
    """메인 파이프라인"""
    temp, co2, oni = load_data()  # 데이터 불러오기
    
    temp.info()
    co2.info()
    oni.info()

    visualize(temp, co2, oni)


def load_data():
    """데이터 불러오기"""
    try:
        co2 = pd.read_csv(BASE / 'data/co2.csv', encoding='utf-8-sig')
        oni = pd.read_csv(BASE / 'data/oni.csv', encoding='utf-8-sig')
        temp = pd.read_csv(BASE / 'data/temp.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        temp, co2, oni = preprocess.preprocess(BASE)

    temp['date'] = pd.to_datetime(temp['date'])
    temp = temp[temp['date'].dt.year <= 2025]
    
    return temp, co2, oni


def visualize(temp: pd.DataFrame, co2: pd.DataFrame, oni: pd.DataFrame):
    """데이터 시각화"""
    # 연도별 수능 기간(11월 8~21일) 최저 기온 평균
    temp_yr = temp.groupby(temp['date'].dt.year)['temp_min'].mean()
    plt.plot(temp_yr.index, temp_yr, marker='o')
    plt.xlabel('연도')
    plt.ylabel('11월 8~21일 평균 최저 기온 (℃)')
    plt.title('수능 시기 서울 평균 최저 기온')
    plt.grid(alpha=0.3)
    plt.savefig(BASE / 'figures/temp_yr_trend.svg', bbox_inches='tight')
    plt.close()

    plt.figure()

    # ONI와 연도별 수능 기간 최저 기온 평균 산점도
    aso = oni.set_index('year')['aso'].loc[temp_yr.index]   # temp_yr과 같은 연도만, 같은 순서로

    fig, ax = plt.subplots()
    ax.scatter(aso, temp_yr, color='black', zorder=3)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)

    ax.axvspan(xmin, -0.5, color='#4a7fd4', alpha=0.12, label='라니냐')
    ax.axvspan(-0.5, 0.5, color='#999999', alpha=0.12, label='평상시')
    ax.axvspan(0.5, xmax, color='#d9534f', alpha=0.12, label='엘니뇨')

    ax.axvline(-0.5, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1)

    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlabel('ONI (8~10월)')
    ax.set_ylabel('수능 시기 평균 최저기온 (℃)')
    fig.savefig(BASE / 'figures/scatter_oni.svg', bbox_inches='tight')
    plt.close(fig)

    plt.figure()

    # 연도별 CO2 농도 변화
    plt.plot(co2['year'], co2['co2'], marker='o')
    plt.xlabel('연도')
    plt.ylabel('이산화 탄소 농도 (ppm)')
    plt.title('연도별 이산화 탄소 농도 변화')
    plt.grid(alpha=0.3)
    plt.savefig(BASE / 'figures/co2_trend.svg', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()