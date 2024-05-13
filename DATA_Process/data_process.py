import os
import pandas as pd


def classify_vehicle_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path, low_memory=False)

    # 判断是否为电车数据
    is_electric_vehicle = (
            df['Short Term Fuel Trim Bank 1[%]'].isnull() &
            df['Short Term Fuel Trim Bank 2[%]'].isnull() &
            df['Long Term Fuel Trim Bank 1[%]'].isnull() &
            df['Long Term Fuel Trim Bank 2[%]'].isnull() &
            (df['Engine RPM[RPM]'].isnull() | (df['Engine RPM[RPM]'] == 0)) &
            (df['Absolute Load[%]'].isnull() | (df['Absolute Load[%]'] == 0))
    )

    # 分离电车数据和油车数据
    electric_vehicle_data = df[is_electric_vehicle]
    gas_vehicle_data = df[~is_electric_vehicle]

    return electric_vehicle_data, gas_vehicle_data


def merge_and_save_data(data_list, output_file):
    # 合并数据
    merged_data = pd.concat(data_list)

    # 保存数据到CSV文件
    merged_data.to_csv(output_file, index=False)


def process_folder(folder_path, output_folder):
    # 遍历文件夹下的所有CSV文件
    electric_vehicle_data_list = []
    gas_vehicle_data_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            electric_vehicle_data, gas_vehicle_data = classify_vehicle_data(file_path)

            electric_vehicle_data_list.append(electric_vehicle_data)
            gas_vehicle_data_list.append(gas_vehicle_data)

    # 合并电车数据和油车数据并保存
    electric_path = os.path.join(output_folder, "electric_vehicle_data.csv")
    gas_path = os.path.join(output_folder, "gas_vehicle_data.csv")
    merge_and_save_data(electric_vehicle_data_list, electric_path)
    merge_and_save_data(gas_vehicle_data_list, gas_path)
    return electric_path, gas_path


def reprocessE(electric_csv):
    # 读取已有的电车数据 CSV 文件
    df = pd.read_csv(electric_csv, low_memory=False)

    # 删除不需要的列
    df = df.drop(columns=['DayNum', 'Matched Longitude[deg]', 'Matchted Latitude[deg]',
                          'Match Type', 'Air Conditioning Power[kW]', 'Timestamp(ms)',
                          'Short Term Fuel Trim Bank 1[%]', 'Long Term Fuel Trim Bank 1[%]',
                          'Long Term Fuel Trim Bank 2[%]', 'Speed Limit[km/h]'])

    # 合并 'Focus Points' 和 'Focus Points;' 列数据
    df['Focus Points'] = df['Focus Points'].fillna(df['Focus Points;'])

    # 删除 'Focus Points;' 列
    df = df.drop(columns=['Focus Points;'])

    # 将 'Energy_Consumption' 移动到最后一列
    energy_consumption = df.pop('Energy_Consumption')
    df.insert(len(df.columns), 'Energy_Consumption', energy_consumption)

    # 根据条件重新判断是否为电车数据
    is_electric_vehicle = (
            (df['Fuel Rate[L/hr]'].isnull() | (df['Fuel Rate[L/hr]'] == 0)) &
            df['HV Battery Current[A]'].notnull() &
            df['HV Battery Voltage[V]'].notnull() &
            df['HV Battery SOC[%]'].notnull()
    )
    df = df[is_electric_vehicle]

    # 保存处理后的数据到同一 CSV 文件
    df.to_csv(electric_csv, index=False)


def reprocessG(gas_csv):
    df = pd.read_csv(gas_csv, low_memory=True)

    # 删除不需要的列
    df = df.drop(columns=['DayNum', 'Matched Longitude[deg]', 'Matchted Latitude[deg]',
                          'Match Type', 'Air Conditioning Power[kW]', 'Timestamp(ms)',
                          'Short Term Fuel Trim Bank 1[%]', 'Long Term Fuel Trim Bank 1[%]',
                          'Long Term Fuel Trim Bank 2[%]', 'Speed Limit[km/h]', 'Focus Points;'])

    is_gas_vehicle = (
            (df['Fuel Rate[L/hr]'].notnull()) &
            df['HV Battery Current[A]'].isnull() &
            df['HV Battery Voltage[V]'].isnull() &
            df['HV Battery SOC[%]'].isnull()
    )
    df = df[is_gas_vehicle]

    # 保存处理后的数据到同一 CSV 文件
    df.to_csv(gas_csv, index=False)


# 合并csv文件的函数
def combine_csv(input_folder, output_folder, file_name):
    os.makedirs(output_folder, exist_ok=True)
    df_list = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    save_path = os.path.join(output_folder, file_name)
    combined_df.to_csv(save_path, index=False)
    return save_path


# 在合并后的csv文件中单独进行数据处理
def process_combined_csv(data_path, output_folder, file_name):
    df = pd.read_csv(data_path)
    # 删除不需要的列
    columns = ['DayNum', 'Timestamp(ms)', 'Matchted Latitude[deg]', 'Matched Longitude[deg]',
               'Match Type', 'Air Conditioning Power[kW]']
    df = df.drop(columns=columns)
    save_path = os.path.join(output_folder, file_name)
    df.to_csv(save_path, index=False)
    return save_path


if __name__ == '__main__':
    """
        把所有的原始数据week.csv放在DATA_Process/DATA文件夹下面，输出的CSV文件统一放在Output文件夹下面
    """
    input_folder = "./DATA"
    output_folder = "./Output"
    # 1.所有的week.csv合并
    save_path = combine_csv(input_folder, output_folder, "combined_data.csv")
    # 2.删除所有合并之后不需要的数据（在电车和油车中都不需要）
    save_path = process_combined_csv(save_path, output_folder, "processed_data.csv")
    # 3.


    # electric_path, gas_path = process_folder(input_folder, output_folder)
    # reprocessE(electric_path)
    # reprocessG(gas_path)
