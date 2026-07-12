def review_data(data, name):

    print("="*60)
    print(name)
    print("="*60)

    print("\nShape")
    print(data.shape)

    print("\nColumns")
    print(data.columns.tolist())

    print("\nData Types")
    print(data.dtypes)

    print("\nMissing Values")
    print(data.isnull().sum())

    print("\nDuplicates")
    print(data.duplicated().sum())

    print("\nFirst Five Rows")
    display(data.head())