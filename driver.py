import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
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


def monte_carlo(data):
    # Number of simulations will equal length of dataset (number of rows)
    num_simulations = len(data)

    def generate_random_numbers(mean, sd):
        random_nums = norm.rvs(loc=mean,
                               scale=sd,
                               size=num_simulations)
        return random_nums

    # Target variable is MntWines, obtaining median and sd
    mnt_wines_expected = data[['MntWines']].mean()
    mnt_wines_sd = data[['MntWines']].std()

    wine_spent = generate_random_numbers(mnt_wines_expected, mnt_wines_sd)

    df = pd.DataFrame(columns=["MntWineSpent"])

    plt.hist(wine_spent, bins='auto')
    # plt.show()
    for i in range(num_simulations):
        dictionary = {"MntWineSpent": round(wine_spent[i], 2)}
        df = df.append(dictionary, ignore_index=True)

    data = pd.concat([data, df], axis=1, join="inner")
    return data


def main():
    df = pd.read_csv("marketing_data.csv", sep=",")
    df = prepare_df(df)
    df = monte_carlo(df)


if __name__ == "__main__":
    main()
