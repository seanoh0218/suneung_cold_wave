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
    temp, co2, oni, suneung_temp = load_data()  # 데이터 불러오기
    
    temp.info()
    co2.info()
    oni.info()

    visualize(temp, co2, oni, suneung_temp)


def load_data():
    """데이터 불러오기"""
    try:
        co2 = pd.read_csv(BASE / 'data/co2.csv', encoding='utf-8-sig')
        oni = pd.read_csv(BASE / 'data/oni.csv', encoding='utf-8-sig')
        temp = pd.read_csv(BASE / 'data/temp.csv', encoding='utf-8-sig')
        suneung_temp = pd.read_csv(BASE / 'data/suneung_temp.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        temp, co2, oni = preprocess.preprocess(BASE)
        suneung_temp = preprocess.make_suneung_temp(BASE)

    temp['date'] = pd.to_datetime(temp['date'])
    temp = temp[temp['date'].dt.year <= 2025]
    
    return temp, co2, oni, suneung_temp


def visualize(temp: pd.DataFrame, co2: pd.DataFrame, oni: pd.DataFrame, suneung_temp: pd.DataFrame):
    """데이터 시각화"""
    # 연도별 수능 기간(11월 8~21일) 최저 기온 평균
    temp_yr = temp.groupby(temp['date'].dt.year)['temp_min'].mean()
    plt.plot(temp_yr.index, temp_yr, marker='o', label='관측값')
    a, b = np.polyfit(temp_yr.index, temp_yr, 1)
    plt.plot(temp_yr.index, a * temp_yr.index + b, color='#d9534f',
             linestyle='--', label=f'추세선 ({a * 10:+.2f} ℃/10년)')
    plt.xlabel('연도')
    plt.ylabel('11월 8~21일 평균 최저 기온 (℃)')
    plt.title('수능 시기 서울 평균 최저 기온')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.savefig(BASE / 'figures/temp_yr_trend.svg', bbox_inches='tight')
    plt.close()

    # ONI와 연도별 수능 기간 최저 기온 평균 산점도
    aso = oni.set_index('year')['aso'].loc[temp_yr.index]   # temp_yr과 같은 연도만, 같은 순서로

    fig, ax = plt.subplots()
    ax.scatter(aso, temp_yr, color='black', zorder=3)
    a, b = np.polyfit(aso, temp_yr, 1)
    ax.plot(aso.sort_values(), a * aso.sort_values() + b,
            color='#7b3fa0', linestyle='--', linewidth=1.5, label='추세선')
    r = np.corrcoef(aso, temp_yr)[0, 1]
    ax.text(0.03, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)

    ax.axvspan(xmin, -0.5, color='#4a7fd4', alpha=0.12, label='라니냐')
    ax.axvspan(-0.5, 0.5, color='#999999', alpha=0.12, label='평상시')
    ax.axvspan(0.5, xmax, color='#d9534f', alpha=0.12, label='엘니뇨')

    ax.axvline(-0.5, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1)

    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlabel('ONI (8~10월)')
    ax.set_ylabel('수능 시기 평균 최저 기온 (℃)')
    ax.set_title('ONI와 수능 시기 평균 최저 기온')
    fig.savefig(BASE / 'figures/scatter_oni.svg', bbox_inches='tight')
    plt.close(fig)

    # 엘니뇨·라니냐·평상시 해별 수능 시기 평균 최저 기온 비교
    phase = pd.Series(np.select([aso <= -0.5, aso >= 0.5], ['라니냐', '엘니뇨'], default='평상시'),
                      index=aso.index)
    order = ['라니냐', '평상시', '엘니뇨']
    df = pd.DataFrame({'phase': phase, 'temp': temp_yr})
    print(df.groupby('phase')['temp'].describe().loc[order].round(2))

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='phase', y='temp', order=order, hue='phase', hue_order=order,
                palette=['#4a7fd4', '#999999', '#d9534f'], boxprops={'alpha': 0.5},
                showfliers=False, legend=False, ax=ax)
    np.random.seed(42)
    sns.stripplot(data=df, x='phase', y='temp', order=order, color='black', size=5, jitter=0.1, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('수능 시기 평균 최저 기온 (℃)')
    ax.set_title('엘니뇨·라니냐 해별 수능 시기 평균 최저 기온')
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(BASE / 'figures/box_enso.svg', bbox_inches='tight')
    plt.close(fig)

    # 연도별 CO2 농도 변화
    plt.plot(co2['year'], co2['co2'], marker='o', label='관측값')
    a, b = np.polyfit(co2['year'], co2['co2'], 1)
    plt.plot(co2['year'], a * co2['year'] + b, color='#d9534f',
             linestyle='--', label=f'추세선 ({a * 10:+.1f} ppm/10년)')
    plt.xlabel('연도')
    plt.ylabel('이산화 탄소 농도 (ppm)')
    plt.title('연도별 이산화 탄소 농도 변화')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.savefig(BASE / 'figures/co2_trend.svg', bbox_inches='tight')
    plt.close()

    # 실제 수능 당일 최저 기온과 수능 기간으로 잡은 기간(11월 8~21일) 사이의 상관관계
    y = suneung_temp.set_index('year')['temp_min']
    x = temp_yr.loc[y.index]

    plt.scatter(x, y, marker='o')
    a, b = np.polyfit(x, y, 1)
    plt.plot(x.sort_values(), a * x.sort_values() + b, color='#d9534f')
    r = np.corrcoef(x, y)[0, 1]
    plt.text(0.03, 0.95, f'r = {r:.2f}', transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel('수능 시기 평균 최저 기온 (℃)')
    plt.ylabel('실제 수능 날의 최저 기온 (℃)')
    plt.title('수능 당일 최저 기온과 수능 시기 평균 최저 기온')
    plt.grid(alpha=0.3)
    plt.savefig(BASE / 'figures/scatter_suneung.svg', bbox_inches='tight')
    plt.close()

    # 수능 기간으로 잡은 기간(11월 8~21일)과 
    y = suneung_temp.set_index('year')['temp_min']
    x = temp_yr.loc[y.index]

    plt.scatter(x, y, marker='o')
    a, b = np.polyfit(x, y, 1)
    plt.plot(x.sort_values(), a * x.sort_values() + b, color='#d9534f')
    r = np.corrcoef(x, y)[0, 1]
    plt.text(0.03, 0.95, f'r = {r:.2f}', transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlabel('수능 시기 평균 최저 기온 (℃)')
    plt.ylabel('실제 수능 날의 최저 기온 (℃)')
    plt.title('수능 당일 최저 기온과 수능 시기 평균 최저 기온')
    plt.grid(alpha=0.3)
    plt.savefig(BASE / 'figures/scatter_suneung.svg', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()