import os
import pandas as pd
import numpy as np


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
    electric_path =  os.path.join(output_folder, "electric_vehicle_data.csv")
    gas_path = os.path.join(output_folder, "gas_vehicle_data.csv")
    merge_and_save_data(electric_vehicle_data_list,electric_path)
    merge_and_save_data(gas_vehicle_data_list, gas_path)
    return  electric_path,gas_path


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
                          'Long Term Fuel Trim Bank 2[%]', 'Speed Limit[km/h]','Focus Points;'])


    is_gas_vehicle = (
            (df['Fuel Rate[L/hr]'].notnull()) &
            df['HV Battery Current[A]'].isnull() &
            df['HV Battery Voltage[V]'].isnull() &
            df['HV Battery SOC[%]'].isnull()
    )
    df = df[is_gas_vehicle]

    # 保存处理后的数据到同一 CSV 文件
    df.to_csv(gas_csv, index=False)



if __name__ =='__main__':
    # 指定文件夹路径和输出文件夹路径
    folder_path = "D:\数模\统计大赛\eved_dataset\data\eVED"
    output_folder = "D:\python\SMC\DATA\Output"
    # 处理文件夹中的CSV文件
    electric_path, gas_path = process_folder(folder_path, output_folder)
    reprocessE(electric_path)
    reprocessG(gas_path)


