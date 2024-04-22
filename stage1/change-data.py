import pandas as pd

# Read the CSV file
df = pd.read_csv("D:\python\SMC\stage1\output_total.csv")

# Get the 'Energy_Consumption' column
energy_consumption = df.pop('Energy_Consumption')

# Append the 'Energy_Consumption' column to the DataFrame
df['Energy_Consumption'] = energy_consumption

# Write the modified DataFrame to a new CSV file
df.to_csv("modified_file.csv", index=False)
