import pandas as pd

if __name__ == "__main__":
    input_csv = 'output_null.csv'
    output_csv = 'origin_data.csv'

    data = pd.read_csv(input_csv)

    # 筛选出符合条件的数据
    filtered_data = data[
        (data['Engine RPM[RPM]'] == 0) &
        (data['Fuel Rate[L/hr]'] == 0) &
        (data['Air Conditioning Power[kW]'].notnull()) &
        (data['Speed Limit[km/h]'].notnull()) &
        (data['Speed Limit[km/h]'] != "48-72")  # 去除值为 "48-72" 的行
        ]

    # 填充空值
    filtered_data['Intersection'].fillna(0, inplace=True)
    filtered_data['Bus Stops'].fillna(0, inplace=True)
    filtered_data['Focus Points'].fillna(0, inplace=True)
    filtered_data['Gradient'].fillna(0, inplace=True)

    # 根据值替换Focus Points列的空白和特定值
    filtered_data['Focus Points'].replace({'crossing': 1, 'stop': 2, 'traffic_signals': 3}, inplace=True)

    # 将筛选后的数据保存到新的CSV文件中
    filtered_data.to_csv(output_csv, index=False)

    print("筛选后的数据已保存到", output_csv)

    df = pd.read_csv("origin_data.csv")
    energy_consumption = df.pop('Energy_Consumption')
    df['Energy_Consumption'] = energy_consumption
    df.to_csv("modified_file.csv", index=False)