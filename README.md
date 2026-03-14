# Enterprise Demand & Workforce Forecasting (Distributed Prophet)

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![Apache Spark](https://img.shields.io/badge/PySpark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Time Series](https://img.shields.io/badge/Machine_Learning-Time_Series_Forecasting-0194E2?style=for-the-badge)

## Executive Summary

This project implements an enterprise-grade Time-Series Forecasting engine on **Databricks** to predict daily operational demand (call volumes) across multiple business departments. 

By leveraging **Facebook Prophet** and **PySpark Pandas UDFs**, this architecture bypasses standard single-node limitations, enabling the simultaneous, parallel training of independent forecasting models. This framework empowers operations teams to:

* **Optimize Agent Allocation:** Move from reactive staffing to highly precise, data-driven workforce scheduling.
* **Reduce Operational Expenditure (OpEx):** Minimize overstaffing during lulls while protecting service level agreements (SLAs) during peak demand.
* **Scale Forecasting:** Train hundreds of departmental forecasts in minutes using distributed cloud computing.

## The Tech Stack

* **Core Logic:** Python, PySpark SQL, Pandas
* **Distributed Computing:** PySpark Pandas UDFs (`applyInPandas`), Apache Arrow
* **Modeling & Evaluation:** Facebook Prophet (Time-Series with daily/weekly/yearly seasonality)
* **Cloud Infrastructure:** Databricks, Unity Catalog (Delta Tables)

## Key Architecture & Business Impact

### The "Pandas UDF" Advantage
Traditional forecasting requires sequential loops (training department 1, then department 2). This project demonstrates **Senior-level Data Engineering** by wrapping the Prophet training logic inside a PySpark Pandas User-Defined Function (UDF). By grouping the data by department and utilizing `.applyInPandas()`, the Databricks cluster distributes the workload across worker nodes, training all models simultaneously in parallel. 

### Simulated Business Logic
The data generation script accurately mimics complex real-world call center behaviors, including:
* **Day-of-Week Seasonality:** Heavy Monday volume spikes vs. Weekend drops.
* **Monthly/Holiday Seasonality:** Surges during Winter/Holiday months.
* **Macro Business Trends:** Year-over-year organic growth.

## Repository Structure

* **`01_call_volume_simulation.py`**: A PySpark data engineering script utilizing the `explode(sequence())` hack to generate 3 continuous years of daily synthetic call volume data across multiple departments, saving to Unity Catalog.
* **`02_distributed_prophet.py`**: The core ML pipeline. Ingests historical data, defines the strict PySpark `StructType` schema, distributes training across the cluster via Pandas UDFs, and saves the 365-day future forecast to Delta Tables.

## How to Run This Project

1. Attach the scripts to an active Databricks Compute Cluster.
2. **Crucial:** Ensure `prophet` is installed at the cluster level (Compute -> Configuration -> Dependencies -> `prophet`).
3. Ensure you have access to `workspace.default` in Unity Catalog.
4. Run `01_call_volume_simulation.py` to generate the Delta table.
5. Run `02_distributed_prophet.py` to distribute the training and view the 2025 forecasts.