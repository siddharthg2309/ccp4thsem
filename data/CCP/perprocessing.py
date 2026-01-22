import pandas as pd

RAIN_FILE = "POWER_Point_Monthly_20000101_20231231_013d08N_080d28E_UTC.csv"
GW_FILE = "Hydrograph_Data_W130130080153001.csv"
OUTPUT_FILE = "final_processed_groundwater_dataset.csv"

# -------------------------------
# READ RAINFALL DATA
# -------------------------------
with open(RAIN_FILE, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith("PARAMETER,YEAR"):
        header_row = i
        break

rain_df = pd.read_csv(RAIN_FILE, skiprows=header_row)

rain_df = rain_df[rain_df.iloc[:,0] == "IMERG_PRECTOT"]

rain_df = rain_df.drop(columns=[c for c in rain_df.columns if c in ["PARAMETER","ANN"]])

rain_df = rain_df.melt(
    id_vars=["YEAR"],
    var_name="MONTH_NAME",
    value_name="RAINFALL"
)

month_map = {
    "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
    "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12
}

rain_df["MONTH"] = rain_df["MONTH_NAME"].map(month_map)
rain_df = rain_df.dropna()

rain_df = rain_df[["YEAR","MONTH","RAINFALL"]]

# -------------------------------
# READ GROUNDWATER DATA
# -------------------------------
gw_df = pd.read_csv(GW_FILE)

gw_df.rename(columns={"Date":"DATE","Water Level":"GW_LEVEL"}, inplace=True)
gw_df["DATE"] = pd.to_datetime(gw_df["DATE"])

gw_df["YEAR"] = gw_df["DATE"].dt.year
gw_df["MONTH"] = gw_df["DATE"].dt.month

gw_df = gw_df[(gw_df["YEAR"] >= 2000) & (gw_df["YEAR"] <= 2023)]

gw_monthly = (
    gw_df.groupby(["YEAR","MONTH"], as_index=False)
    .agg({"GW_LEVEL":"mean"})
)

# -------------------------------
# MERGE (CORRECT WAY)
# -------------------------------
df = pd.merge(
    gw_monthly,
    rain_df,
    on=["YEAR","MONTH"],
    how="inner"
)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["DATE"] = pd.to_datetime(
    df["YEAR"].astype(str) + "-" +
    df["MONTH"].astype(str).str.zfill(2) + "-01"
)

def season(m):
    if m in [12,1,2]:
        return 1
    elif m in [3,4,5]:
        return 2
    elif m in [6,7,8,9]:
        return 3
    else:
        return 4

df["SEASON"] = df["MONTH"].apply(season)

df = df.sort_values("DATE")

df["GW_LAG_1"] = df["GW_LEVEL"].shift(1)
df["GW_LAG_3"] = df["GW_LEVEL"].shift(3)
df["GW_LAG_6"] = df["GW_LEVEL"].shift(6)

df = df.dropna()

df = df[
    ["DATE","GW_LEVEL","RAINFALL","MONTH","SEASON","GW_LAG_1","GW_LAG_3","GW_LAG_6"]
]

df.to_csv(OUTPUT_FILE, index=False)

print("✅ Final dataset saved:", OUTPUT_FILE)
print("📊 Total rows:", len(df))
