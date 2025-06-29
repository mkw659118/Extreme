import numpy as np
import pandas as pd
import pickle 

def get_sql_format_data(current_date, pred_value, cleaned_input):
    from datetime import datetime
    now_df = []
    cleaned_input = cleaned_input[0, :, :]
    for j in range(pred_value.shape[1]):
        idx = np.random.randint(0, 10)  # 生成一个 0 到 9（包含 0，不包含 10）之间的整数
        fund_code = cleaned_input[j][0]
        forcast_date = current_date
        pred = '{"pre": [' + ', '.join(f'{item:.6f}' for item in pred_value[:, j]) + ']}'
        model_version = 'v2025'
        create_date = update_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        now_df.append([idx, fund_code, forcast_date, pred, model_version, create_date, update_date])
        # break
    # now_df
    now_df = np.array(now_df)
    now_df = pd.DataFrame(now_df, columns=['id', 'fund_code', 'forecast_date', 'pre_data', 'model_version',
       'create_time', 'update_time'])
    return now_df

def insert_pred_to_sql(df, table_name):
    try:
        # 读取数据库的访问权限token代码
        with open('./datasets/sql_token.pkl', 'rb') as f:
            DB_URI = pickle.load(f)
        # 这个是数据库引擎代码
        engine = create_engine(DB_URI)
        # 插入数据
        df.to_sql(
            name=table_name,   # 表名
            con=engine,                   # 数据库连接
            if_exists='append',           # 追加到已有表中
            index=False                   # 不插入索引列
        )
        print(f"✅ 数据成功写入数据库{table_name}。")

    except FileNotFoundError:
        print("❌ 无法找到 sql_token.pkl 文件。请检查路径是否正确。")

    except SQLAlchemyError as e:
        print(f"❌ 数据库插入失败: {e}")

    except Exception as e:
        print(f"❌ 发生未知错误: {e}")