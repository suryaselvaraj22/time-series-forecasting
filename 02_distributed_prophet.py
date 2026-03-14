# 02_distributed_prophet.py
# Objective: Train a Time-Series forecasting model for each Call Center Department.
# We use PySpark Pandas UDFs to train all department models simultaneously in parallel across the cluster!

# Tried for fun: We can install packages on the Driver Node, but it's useless here because the worker nodes won't have Prophet anyway!
#import subprocess
#import sys

#try:
    #import prophet
    #print("Prophet is already installed!")
#except ImportError:
    #print("Prophet not found. Installing now...")
    #subprocess.check_call([sys.executable, "-m", "pip", "install", "prophet"])
    #print("Prophet installation complete!")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, rand, randn, round, exp, abs, avg
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType
import pandas as pd 

spark = SparkSession.builder.appName("Distributed_Prophet_Forecasting").getOrCreate()
print("Starting Distributed Prophet Forecasting...") 

# 1. Load the Historical Call Volume Data
input_table = "workspace.default.concentrix_call_volume"  
print(f"Loading data from {input_table}...")
df_history = spark.table(input_table)

# 2. Define the Output Schema for the PySpark Cluster
# When the worker nodes finish training and predicting in Pandas, they need to know 
# exactly what format PySpark expects the data to be in when they hand it back.
result_schema = StructType([
    StructField("department", StringType(), True),  # Department name 
    StructField("ds", DateType(), True),  # Date of the forecasted value
    StructField("yhat", DoubleType(), True),  # The forecasted call volume
    StructField("yhat_lower", DoubleType(), True),  # Lower Confidence Interval
    StructField("yhat_upper", DoubleType(), True)  # Upper Confidence Interval
])

# 3. Define the Python/Pandas function that will run on EACH worker node
# This function receives a single Pandas DataFrame containing exactly ONE department's history.
def forecast_department(history_pd: pd.DataFrame) -> pd.DataFrame:
    # Again, tried for fun: As expected Worker Nodes can't install packages because they don't have sudo privileges! 
    # import subprocess
    # import sys

    # WORKER NODE H*A*C*K: Ensure Prophet is installed on this specific worker node
    # try:
    #     import prophet
    #     print("Prophet is already installed!")
    # except ImportError:
    #     print("Prophet not found. Installing now...")
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "prophet"])
    #     print("Prophet installation complete!")

    #Please ensure Prophet is installed on your cluster before running this code, otherwise it will fail!
    # Now that Prophet is installed via the Cluster UI, we can just import it normally!
    import pandas as pd
    try: 
        from prophet import Prophet  

        # Initialize the Prophet model, explicitly turning on seasonality
        m = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            daily_seasonality=False # Not predicting hourly data
        )  
        # Fit the model to this specific department's historical data
        m.fit(history_pd)

        # Create a future dataframe projecting 365 days (1 year) into the future
        future_pd = m.make_future_dataframe(periods=365, freq='D')

        # Generate the forecast
        forecast_pd = m.predict(future_pd)

        # Extract the department name from the historical data
        dept_name = history_pd['department'].iloc[0]

        # Format the results to match our PySpark result_schema perfectly
        result_pd = forecast_pd[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result_pd['department'] = dept_name  # Add the department column 

        # B*U*G FIX: Prophet outputs timestamps, but PySpark expects Dates. 
        # If we don't cast this, PySpark's Arrow stream collapses and returns "NoneType"
        result_pd['ds'] = pd.to_datetime(result_pd['ds']).dt.date
        result_pd['yhat'] = result_pd['yhat'].astype(float)
        result_pd['yhat_lower'] = result_pd['yhat_lower'].astype(float)
        result_pd['yhat_upper'] = result_pd['yhat_upper'].astype(float)

        # Reorder columns to match schema: department, ds, yhat, yhat_lower, yhat_upper
        return result_pd[['department', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    except Exception as e:
        # ULTIMATE DEBUGGER H*A*C*K: If it crashes, return the error AS the data!
        error_df = pd.DataFrame({
            'department': [f"ERROR: {str(e)}"],
            'ds': [pd.to_datetime('2025-01-01').date()],  
            'yhat': [-1.0],
            'yhat_lower': [-1.0],
            'yhat_upper': [-1.0]
        })
        return error_df

# 4. Execute the Distributed Training!
print("\nDistributing Prophet training across the cluster... (Predicting 365 days into 2025)")

# This is the magic PySpark line: Group by department, send to workers, run the UDF, collect results.
df_forecast = df_history.groupBy("department").applyInPandas(
    forecast_department, 
    schema=result_schema
)

# 5. Save the Forecasts to Unity Catalog
output_table = "workspace.default.concentrix_volume_forecast"
print(f"\nSaving 2025 Forecast to Unity Catalog: {output_table}...")

df_forecast.write.format("delta").mode("overwrite").saveAsTable(output_table)

print("\n✅ Distributed Forecasting Complete!")

# Display the forecasted rows for the year 2025 to see future predictions
display(df_forecast.filter(df_forecast.ds >= "2025-01-01").orderBy("department", "ds").limit(15))


