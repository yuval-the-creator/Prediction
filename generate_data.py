import pandas as pd
import numpy as np

# Parameters
num_departments = 5
months = pd.date_range(end="2025-09-01", periods=60, freq="M")
departments = [f"dep_{i+1}" for i in range(num_departments)]

# Generate random employee counts for each department over time
data = []
for dep in departments:
    type1_employees = np.random.randint(100, 1000, size=len(months))
    type2_employees = np.random.randint(20, 500, size=len(months))
    data.append(
        pd.DataFrame(
            {
                "department": dep,
                "month": months,
                "type1_employees": type1_employees,
                "type2_employees": type2_employees,
            }
        )
    )

# Combine all departments into one DataFrame
df = pd.concat(data, ignore_index=True)
df = df.set_index(["department", "month"]).sort_index()

# create history columns for each department
for lag in range(1, 13):  # 1..12
    df[f"type1_hist_{lag}"] = df.groupby(level=0)["type1_employees"].shift(lag)
    df[f"type2_hist_{lag}"] = df.groupby(level=0)["type2_employees"].shift(lag)

df.dropna(inplace=True)
