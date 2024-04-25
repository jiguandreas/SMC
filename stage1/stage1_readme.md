1.原始数据集包含以下变量：
    DayNum: 数据记录日期
    VehId: 车辆的标识
    Trip: 行驶路段的编号
    Timestamp(ms): 时间点
    Latitude[deg]: 纬度
    Longitude[deg]: 经度
    Vehicle Speed[km/h]: 车速
    MAF[g/sec]: 进入**发动机**的空气量
    Engine RPM[RPM]: 发动机转速
    Absolute Load[%]: 发动机绝对负荷
    OAT[DegC]: 外部气温
    Fuel Rate[L/hr]: 燃油消耗率
    Air Conditioning Power[kW]: 以千瓦为单位的空调功率
    Air Conditioning Power[Watts]: 以瓦特为单位的空调功率
    Heater Power[Watts]: 加热器功率
    HV Battery Current[A]: 电池的充放电电流
    HV Battery SOC[%]: 剩余电量百分比
    HV Battery Voltage[V]: 电池电压
    Short Term Fuel Trim Bank 1[%]: 内燃机调节空燃
    Short Term Fuel Trim Bank 2[%]: 内燃机调节空燃
    Long Term Fuel Trim Bank 1[%]: 调整燃油
    Long Term Fuel Trim Bank 2[%]: 调整燃油
    Elevation Raw[m]: 海拔高度
    Elevation Smoothed[m]: 平滑后的海拔高度
    Gradient: 路面坡度
    Energy_Consumption: 能量消耗量
    Matchted Latitude[deg]: 校正后纬度
    Matched Longitude[deg]: 校正后经度
    Match Type: 校正的方式
    Class of Speed Limit: 速度限制的类别
    Speed Limit[km/h]: 速度限制
    Speed Limit with Direction[km/h]: 有方向的速度限制
    Intersection: 交叉口
    Bus Stops: 公交站点
    Focus Points: 关注点，nan

2.数据处理：
我们人为剔除掉一些无关的数据：DayNum\Timestamp\Focus Points
判断油车的数据：MAF、Engine RPM、Absolute Load、Fuel Rate
判断电车的数据：HV Battery Current、HV Battery SOC、HV Battery Voltage


3.进行主成分分析或因子分析对数据降维(保留信息要>=90%)，然后进行相关性分析，找出相关性较高的变量，进行变量筛选。

4.使用逻辑回归、决策树、随机森林、XGBoost等算法进行建模，对模型进行评估，选择最优模型。


for step1: MakeData2CSV_byGPT.py
for step2: csv_process.py
for step3: Analysis_factors.py
for step4: EC_Regression_by_factors.py
