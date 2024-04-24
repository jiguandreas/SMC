1.原始数据集包含以下变量：
    DayNum: 表示数据记录的日期编号。
    VehId: 车辆标识符。
    Trip: 行程编号，表示这是车辆某天的第几次行程。
    Timestamp(ms): 时间戳，以毫秒为单位，记录数据的具体时间点。
    Latitude[deg]: 纬度，以度为单位。
    Longitude[deg]: 经度，以度为单位。
    Vehicle Speed[km/h]: 车辆速度，以公里/小时为单位。
    MAF[g/sec]: 空气流量传感器测量的空气质量，以克/秒计。
    Engine RPM[RPM]: 发动机转速，以转/分钟计。
    Absolute Load[%]: 发动机绝对负荷百分比。
    OAT[DegC]: 外部气温，以摄氏度为单位。
    Fuel Rate[L/hr]: 燃油消耗率，以升/小时计。
    Air Conditioning Power[kW]: 空调功率，以千瓦为单位。
    Air Conditioning Power[Watts]: 空调功率，以瓦特为单位。
    Heater Power[Watts]: 加热器功率，以瓦特为单位。
    HV Battery Current[A]: 高压电池电流，以安培为单位。
    HV Battery SOC[%]: 高压电池剩余电量百分比。
    HV Battery Voltage[V]: 高压电池电压，以伏特为单位。
    Short Term Fuel Trim Bank 1[%]: 短期燃油调整比例（第一缸组），百分比。
    Short Term Fuel Trim Bank 2[%]: 短期燃油调整比例（第二缸组），百分比。
    Long Term Fuel Trim Bank 1[%]: 长期燃油调整比例（第一缸组），百分比。
    Long Term Fuel Trim Bank 2[%]: 长期燃油调整比例（第二缸组），百分比。
    Elevation Raw[m]: 原始海拔高度，以米为单位。
    Elevation Smoothed[m]: 平滑处理后的海拔高度，以米为单位。
    Gradient: 路面坡度。
    Energy_Consumption: 能量消耗量。
    Matchted Latitude[deg]: 匹配后的纬度。
    Matched Longitude[deg]: 匹配后的经度。
    Match Type: 匹配类型，通常用于指示位置数据的匹配精度。
    Class of Speed Limit: 速度限制的类别。
    Speed Limit[km/h]: 速度限制，以公里/小时为单位。
    Speed Limit with Direction[km/h]: 带方向的速度限制。
    Intersection: 交叉口。
    Bus Stops: 公交站点。
    Focus Points: 关注点，可能是特定的地理或交通特征。

2.我们人为剔除掉一些无关性的数据

2.数据的缺失值、异常值以及标准化处理

3.进行主成分分析或因子分析对数据降维(保留信息要>=90%)，然后进行相关性分析，找出相关性较高的变量，进行变量筛选。

4.使用逻辑回归、决策树、随机森林、XGBoost等算法进行建模，对模型进行评估，选择最优模型。



for step1: MakeData2CSV_byGPT.py
for step2: csv_process.py
for step3: Analysis_factors.py
for step4: EC_Regression_by_factors.py
