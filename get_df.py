from datetime import datetime
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

DB_CONFIG = {
    'host': '123.57.74.222',
    'user': 'data_user',
    'password': 'DataUser123',
    'database': 'ai_data'
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def get_df_from_sql(query: str):
    connection = get_db_connection()
    if not connection:
        raise Exception("Database connection failed")
    try:
        return pd.read_sql(query, connection)
    except Error as e:
        raise Exception(f"Database error: {e}")
    finally:
        if connection.is_connected():
            connection.close()

def date_to_quarter(date_obj):
    quarter = (date_obj.month - 1) // 3 + 1
    return f"{date_obj.year}Q{quarter}"

def get_df(fund_code: str, start_date: str = datetime.now(), end_date: str = datetime.now()):
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    start_quarter = date_to_quarter(start_date)
    end_quarter = date_to_quarter(end_date)

    usdind_df = get_df_from_sql(f"SELECT * FROM usdind WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}' ORDER BY trade_date asc")
    gdp_df = get_df_from_sql(f"SELECT * FROM gdp WHERE quarter BETWEEN '{start_quarter}' AND '{end_quarter}' ORDER BY quarter asc")
    shibor_df = get_df_from_sql(f"SELECT * FROM shibor WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc")
    net_value_df = get_df_from_sql(f"SELECT * FROM net_value WHERE fund_code = '{fund_code}' AND date >= '{start_date}' AND date <= '{end_date}' ORDER BY date asc")

    merged_df = pd.merge(net_value_df, usdind_df, left_on='date', right_on='trade_date', how='left')
    merged_df['quarter'] = merged_df['date'].apply(date_to_quarter)
    merged_df = pd.merge(merged_df, gdp_df, on='quarter', how='left')
    final_df = pd.merge(merged_df, shibor_df, left_on='date', right_on='date', how='left')
    final_df.drop(columns=['trade_date', 'quarter', 'id'], inplace=True, errors='ignore')
    final_df["date"] = pd.to_datetime(final_df["date"])
    final_df.set_index("date", inplace=True)

    # ======================== 替代 empyrical 计算 ========================
    daily_return = pd.DataFrame({
        "date": pd.to_datetime(net_value_df["date"]),
        "daily_return": net_value_df["adj_nav"].pct_change()
    })
    daily_return.set_index("date", inplace=True)

    # 1. cumulative return
    daily_return["cumulative"] = (1 + daily_return["daily_return"]).cumprod() - 1
    final_df["cumulative"] = daily_return["cumulative"]

    # 2. annual volatility
    final_df["annual_volatility"] = daily_return["daily_return"].std() * np.sqrt(252)

    # 3. stability (R² of cumulative return)
    X = np.arange(len(daily_return)).reshape(-1, 1)
    y = daily_return["daily_return"].cumsum().values.reshape(-1, 1)
    if len(y) > 1:
        reg = LinearRegression().fit(X, y)
        final_df["stability"] = reg.score(X, y)
    else:
        final_df["stability"] = np.nan

    # 4. monthly win rate
    monthly_return = daily_return["daily_return"].resample("M").apply(lambda x: (1 + x).prod() - 1)
    final_df["monthwin"] = (monthly_return > 0).sum() / len(monthly_return) if len(monthly_return) > 0 else np.nan

    # 5. daily win rate
    final_df["winning_day"] = (daily_return["daily_return"] > 0).sum() / len(daily_return) if len(daily_return) > 0 else np.nan

    # 6. max drawdown
    cumulative = (1 + daily_return["daily_return"]).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    final_df["maxDrawdown"] = drawdown.min() if not drawdown.empty else np.nan

    return final_df

def get_df_date_as_index(fund_code: str, start_date: str = datetime.now(), end_date: str = datetime.now()):
    return get_df(fund_code, start_date, end_date)

def get_df_date_as_colum(fund_code: str, start_date: str = datetime.now(), end_date: str = datetime.now()):
    df = get_df(fund_code, start_date, end_date)
    df.reset_index(inplace=True)
    return df

if __name__ == "__main__":
    print(get_df_date_as_index("000001", "2020-01-01", "2025-06-30"))
    print(get_df_date_as_colum("000001", "2020-01-01", "2025-06-30"))