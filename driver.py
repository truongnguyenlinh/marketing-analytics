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
    # print(data.head(1))

    # View data-types
    # print(data.dtypes)

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
    # print(data.head(1))

    # Find null values and impute
    # print(data.isnull().sum().sort_values(ascending=False))
    data["Income"] = data["Income"].fillna(data["Income"].median())
    return data


def ab_test(df):
    from math import sqrt
    df_wine = df["MntWines"]
    # df_wine = df_wine.reset_index()
    first_third = df_wine.max() * 0.33
    second_third = df_wine.max() * 0.66
    max_wine = df_wine.max()
    min_wine = df_wine.min()
    df_first = df[df['MntWines'] < first_third]
    df_first = df_first['Income']
    df_second = df[(df['MntWines'] > first_third) & (df['MntWines'] < second_third)]
    df_second = df_second['Income']
    df_third = df[df['MntWines'] > second_third]
    df_third = df_third['Income']

    # H0: Income makes no difference to the amount of wines purchased
    # H1: Wine purchases will increase with an increase in income.

    cohen = ((df_first.mean() - second_third.mean()) / (sqrt((df_first.std() ** 2 + df_second.std() **2) / 2)))
    from statsmodels.stats.power import TTestIndPower
    effect = cohen # Obtained from previous step.
    alpha = 0.05  # Enable 95% confidence for two tail test.
    power = 0.95  # One minus the probability of a type II error.
    # Limits possibility of type II error to 20%.
    analysis = TTestIndPower()
    numSamplesNeeded = analysis.solve_power(effect, power=power, alpha=alpha)
    print(numSamplesNeeded)

    from scipy import stats
    old_menu_sales = [101, 110, 115, 136, 140, 108,
                      80, 89, 131, 98, 121, 117, 106,
                      141, 119, 153, 184, 127, 103,
                      139, 130, 146, 130]

    new_menu_sales = [158, 145, 134, 130, 113, 135,
                      163, 128, 166, 154, 143, 147, 132,
                      132, 136, 99, 163, 106, 143, 168, 136,
                      123, 159]

    testResult = stats.ttest_ind(new_menu_sales,
                                 old_menu_sales, equal_var=False)

    import numpy as np
    print("Hypothesis test p-value: " + str(testResult))
    print("New sales mean: " + str(np.mean(new_menu_sales)))
    print("New sales std: " + str(np.std(new_menu_sales)))


def main():
    df_original = pd.read_csv("marketing_data.csv", sep=",")
    df = prepare_df(df_original)
    ab_test(df)


if __name__ == "__main__":
    main()
