# 01_call_volume_simulation.py
# Objective: Generate 3 years of daily synthetic call center volume data to demonstrate
# Time-Series Forecasting (Prophet) for workforce and agent allocation.


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, rand, randn, round, exp, abs, avg, explode, sequence, to_date, lit, expr, dayofweek, month

spark = SparkSession.builder.appName("Concentrix_Call_Volume_Sim").getOrCreate()
print("Starting Time-Series Call Volume Simulation...")

# 1. Generate a continuous timeline of Dates (Jan 1, 2022 to Dec 31, 2024)
start_date = to_date(lit("2022-01-01"))
end_date = to_date(lit("2024-12-31"))

# Create a single row dataframe with an array of all dates, then explode it into rows
df_dates = spark.createDataFrame([(1,)]).select(
    explode(sequence(start_date, end_date, expr("interval 1 day"))).alias("ds")
)

# 2. Define our Call Center Departments
departments = spark.createDataFrame([
    ("Billing", 1500), # Base volume of 1500 calls/day
    ("Tech Support", 2500), # Base volume of 2500 calls/day
    ("Claims", 800) # Base volume of 800 calls/day
], ["department", "base_volume"]
)
# Cross Join so every date has a row for every department
df_base = df_dates.crossJoin(spark.createDataFrame(departments))

# 3. Engineer Seasonality, Trends, and Noise
#Day of Week Seasonality: Mondays have a massive spike in calls, while weekends are dead.
#Monthly Seasonality: Volume slowly climbs in the winter/holiday months.
#Growth Trend: Call volume slowly trends upward over the 3 years as the "business grows."
print("Injecting business seasonality (Mondays = High, Weekends = Low)...")

df_trends = df_base \
    .withColumn("day_of_week", dayofweek("ds")) \
    .withColumn("month_of_year", month("ds")) \
    .withColumn("trend_growth", (expr("datediff(ds, '2022-01-01')") * 0.25))  # Business grows by 0.25 calls per day over 3 years

# Add Day-of-Week logic (1=Sunday, 2=Monday... 7=Saturday)
# Mondays get a huge 40% spike. Weekends drop by 60%.
df_logic = df_trends.withColumn("dow_multiplier",
    when(col("day_of_week") == 2, 1.40)  # Monday
    .when(col("day_of_week").isin(1, 7), 0.4)  # Weekend drop
    .otherwise(1.0)  # Normal Weekday
)

# Add Monthly Seasonality (Winter/Holidays are busier)
df_logic = df_logic.withColumn("month_multiplier",   
    when(col("month_of_year").isin(11, 12, 1), 1.20)  # Winter surge
    .when(col("month_of_year").isin(6, 7, 8), 0.90)  # Summer lull
    .otherwise(1.0)
)                            

# 4. Calculate Final Daily Volume (y)
# Target format for Prophet is strictly 'ds' (datestamp) and 'y' (value)
print("Calculating final call volumes...")
df_final = df_logic.withColumn("y", 
    round(
        (col("base_volume") + col("trend_growth")) * col("dow_multiplier") * col("month_multiplier") * (1 + (rand() * 0.15 - 0.075))  # Add +/- 7.5% random daily noise
    )
).select("ds", "department", "y").orderBy("ds", "department")

# 5. Save to Unity Catalog
output_table = "workspace.default.concentrix_call_volume" 
print(f"\nSaving generated time-series data to Unity Catalog: {output_table}...")

df_final.write.format("delta").mode("overwrite").saveAsTable(output_table)

print("✅ Data simulation complete! Ready for Prophet Forecasting.")
display(df_final.limit(10))