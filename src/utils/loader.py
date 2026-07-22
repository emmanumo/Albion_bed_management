from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]


DATA_DIR = BASE_DIR / "data" / "processed"



def load_all_data():

    files = {

        "model_df": "bed_occupancy_modelling_dataset.csv",

        "timeseries": "bed_occupancy_timeseries.csv",

        "sarimax_short": "sarimax_24_72_forecast.csv",

        "sarimax_long": "sarimax_7_30_forecast.csv",

        "sarimax_future": "sarimax_future_forecast.csv",

        "xgb_future": "xgb_future_forecast.csv",

        "staffing": "staffing_risk_forecast.csv"

    }


    data = {}

    for name, file in files.items():

        path = DATA_DIR / file

        if path.exists():

            data[name] = pd.read_csv(path)

        else:

            print(f"Missing file: {path}")
            data[name] = pd.DataFrame()


    return data