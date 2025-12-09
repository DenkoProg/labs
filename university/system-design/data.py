import pandas as pd

# 0: Area, 1: Rooms, 2: Price
flats = [[50, 2, 12000], [60, None, 15000], [100, 4, 40000]]
flats = [{"Area": flat[0], "Rooms": flat[1], "Price": flat[2]} for flat in flats]

df = pd.DataFrame(flats)
print(df.describe())

# Handling missing values by filling with median
# df = df.fillna(df.median())
# Handling missing values by dropping rows with any NaN values
# df = df.dropna()
# Manually fixing the missing value for demonstration
df.iloc[1, 1] = 3

print(df.head())

# Basic statistics
print(df["Price"].value_counts())

df.to_csv("flats.csv", index=False)
