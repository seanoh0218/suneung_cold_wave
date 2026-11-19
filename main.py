import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import preprocess
import analysis

BASE = preprocess.BASE

START_YEAR, END_YEAR = 2000, 2025   # 분석 기간 (2026년은 올해 예측 비교용으로 제외)
DAYS_PER_YEAR = 14                  # 11월 8~21일
TRAIN_END = 2019                    # 2000~2019년 학습, 2020~2025년 검증


def main():
    """메인 파이프라인"""
    temp, co2, oni, suneung_temp = load_data()

    # 그래프 그리기 (추세, ONI 산점도, ENSO 상자그림, CO2, 수능 당일 상관)
    analysis.analyze(temp, co2, oni, suneung_temp)

    data = make_dataset(temp, co2, oni)

    # 1) 전체 기간 회귀: 보고서에 쓸 a, b, R²
    print('[전체 기간 회귀 2000~2025]')
    fit_temp_regression(data)

    # 2) 검증: 2000~2019년으로 학습 → 2020~2025년 예측
    print(f'\n[검증 {START_YEAR}~{TRAIN_END} 학습 → {TRAIN_END + 1}~{END_YEAR} 예측]')
    validate_temp_regression(data)


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
    temp = temp[temp['date'].dt.year.between(START_YEAR, END_YEAR)]
    suneung_temp = suneung_temp[suneung_temp['year'].between(START_YEAR, END_YEAR)]

    return temp, co2, oni, suneung_temp


def make_dataset(temp: pd.DataFrame, co2: pd.DataFrame, oni: pd.DataFrame) -> pd.DataFrame:
    """연도별 수능 시기 기온, CO2, ONI(ASO)를 한 표로 합치기"""
    yearly = temp.groupby(temp['date'].dt.year)['temp_min'].agg(['mean', 'count'])
    yearly = yearly.loc[START_YEAR:END_YEAR]
    assert (yearly['count'] == DAYS_PER_YEAR).all(), '14일이 다 안 채워진 연도가 있음'

    data = pd.DataFrame({
        'temp': yearly['mean'],
        'co2': co2.set_index('year')['co2'],
        'oni': oni.set_index('year')['aso'],
    }).loc[yearly.index].rename_axis('year')
    assert data.notna().all().all(), '비어 있는 값이 있음'
    return data


def fit_temp_regression(data: pd.DataFrame) -> LinearRegression:
    """수능 시기 기온 = a·CO2 + b·ONI + 상수 (넘겨받은 기간 전체로 학습)"""
    X = data.drop(columns='temp')
    y = data['temp']

    model = LinearRegression().fit(X, y)

    coef = pd.Series(model.coef_, index=X.columns)
    print(f"a(CO2) = {coef['co2']:.3f} ℃/ppm")
    print(f"b(ONI) = {coef['oni']:.3f} ℃")
    print(f'상수   = {model.intercept_:.2f} ℃')
    print(f'R²     = {model.score(X, y):.3f}')

    return model


def validate_temp_regression(data: pd.DataFrame) -> LinearRegression:
    """학습 기간으로 모델을 만들고 검증 기간을 예측해 실제값과 비교하기"""
    train, test = data.loc[:TRAIN_END], data.loc[TRAIN_END + 1:]
    X_test = test.drop(columns='temp')
    y_test = test['temp']

    # 학습 기간 데이터로만 모델 만들기
    model = fit_temp_regression(train)

    # 검증 기간 기온 예측하기
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)

    # R²와 RMSE로 평가하고, 학습 기간 평균값을 찍었을 때와 비교하기
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    baseline = np.full(len(y_test), train['temp'].mean())
    base_rmse = np.sqrt(mean_squared_error(y_test, baseline))
    print(f'검증 R² = {r2:.3f}')
    print(f'검증 RMSE = {rmse:.1f} ℃ (학습 기간 평균값만 쓴 경우 RMSE = {base_rmse:.1f} ℃)')
    print(pd.DataFrame({'실제': y_test, '예측': y_pred}).round(2))

    # 실제값과 예측값 시각화
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, zorder=3)
    for year in y_test.index:
        ax.annotate(year, (y_test[year], y_pred[year]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], color='#d9534f', linestyle='--', label='실제 = 예측')

    ax.set_xlabel('실제 수능 시기 평균 최저 기온 (℃)')
    ax.set_ylabel('예측 수능 시기 평균 최저 기온 (℃)')
    ax.set_title(f'실제 기온과 예측 기온 ({TRAIN_END + 1}~{END_YEAR}년)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(BASE / 'figures/validation.svg', bbox_inches='tight')
    plt.close(fig)

    return model


if __name__ == '__main__':
    main()