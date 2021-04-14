import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)


def prepare_df(data):
    # View data
    print(data.head(1))

    # View data-types
    print(data.dtypes)

    # Convert categorical data
    data.columns = data.columns.str.strip()
    data["Education"] = data["Education"].replace(["2n Cycle"], "2n_Cycle")
    data["Education"] = data["Education"].astype("category")
    data["Marital_Status"] = data["Marital_Status"].astype("category")
    data["Country"] = data["Country"].astype("object")
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])

    # Encode categorical predictor variables
    categorical_columns = ["Education", "Marital_Status"]
    for cc in categorical_columns:
        dummies = pd.get_dummies(data[cc])
        dummies = dummies.add_prefix("{}#".format(cc))
        data = data.join(dummies)

    # Convert income to int
    data["Income"] = data["Income"].replace({"\$": "", ",": ""}, regex=True)
    data["Income"] = data["Income"].astype("float")

    # Enrollment date
    data["Dt_Year"] = data["Dt_Customer"].dt.year
    data["Dt_Month"] = data["Dt_Customer"].dt.month
    data["Dt_Day"] = data["Dt_Customer"].dt.month

    # View updated dataset
    print(data.head(1))

    # Find null values and impute
    print(data.isnull().sum().sort_values(ascending=False))
    data["Income"] = data["Income"].fillna(data["Income"].median())
    return data


def main():
    df = pd.read_csv("marketing_data.csv", sep=",")
    df = prepare_df(df)


if __name__ == "__main__":
    main()
