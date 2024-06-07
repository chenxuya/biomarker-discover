import pandas as pd
import numpy as np

# 函数：检查DataFrame中的特定值
def check_values(df, value, message):
    # 检查特定值
    is_value = df == value
    if is_value.any().any():
        print(message)
        # 输出特定值所在的行和列的索引
        value_locations = np.where(is_value)
        for row_index, col_index in zip(*value_locations):
            print(f"值{value}位于: 第{df.index[row_index]}行, '{df.columns[col_index]}'列")
        return True
    else:
        print(f"没有发现值{value}。")
        return False
# 检查小于0的值
def check_negative_values(df):
    num_cols = df.select_dtypes(include=[np.number])  # 选择数值型列
    new_df = df.loc[:, num_cols.columns]
    new_df[new_df<0] = np.nan
    is_value = new_df.isnull()
    if is_value.any().any():
        print("存在负值：")
        # 输出负值所在的行和列的索引
        value_locations = np.where(is_value)
        for row_index, col_index in zip(*value_locations):
            print(f"小于0值位于: 第{df.index[row_index]}行, '{df.columns[col_index]}'列")
        return True
    else:
        print("没有发现负值。")
        return False

def check0infna(df, file_name):
    # 假设df是你的DataFrame
    df = read_file(data_path, index_col=0)
    # 检查无穷值
    check_values(df, np.inf, f"{file_name} 存在无穷值：")
    check_values(df, -np.inf, f"{file_name} 存在负无穷值：")

    # 检查空值
    check_values(df, np.nan, f"{file_name} 存在空值：")
    # 检查0值
    # check_values(df, 0, "存在0值：")
    # 调用函数检查小于0的值
    check_negative_values(df)

