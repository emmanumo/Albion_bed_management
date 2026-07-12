def merge_hospital(admissions, hospital):
    return admissions.merge(
        hospital,
        on="hospital_id",
        how="left"
    )