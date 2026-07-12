def occupancy_rate(df):
    df["occupancy_rate"] = (
        df["occupied_beds"] /
        df["staffed_beds"]
    ) * 100

    return df