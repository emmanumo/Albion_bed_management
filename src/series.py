def get_time_series(df, hospital, ward):

    ts = df[
        (df["hospital_id"] == hospital) &
        (df["ward"] == ward)
    ].copy()

    ts = ts.sort_values("date")

    ts.set_index("date", inplace=True)

    return ts