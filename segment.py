import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sqlalchemy import create_engine

# Load samples.
# Load samples.
import pandas as pd

PATH = '/users/pm/desktop/daydocs/data/'
FILE = 'marketing_data.csv'
data = pd.read_csv(FILE)
samples = np.array(data[['MntWines']])


# df_wine = df["MntWines"]
# # df_wine = df_wine.reset_index()
# first_third = df_wine.max() * 0.33
# second_third = df_wine.max() * 0.66
# df_first = df[df['MntWines'] < first_third]
# df_second = df[(df['MntWines'] > first_third) & (df['MntWines'] < second_third)]
# df_second = df_second['Income']
# df_third = df[df['MntWines'] > second_third]


def executeSQL(sql, df):
    # This code creates an in-memory table called 'Inventory'.
    engine = create_engine('sqlite://', echo=False)
    connection = engine.connect()
    df.to_sql(name='Marketing', con=connection, if_exists='replace', index=False)

    # This code performs the query.
    queryResult = pd.read_sql(sql, connection)
    return queryResult


def getPercentileValues(df):
    oneThird = df['MntWines'].quantile(0.333)
    twoThird = df['MntWines'].quantile(0.667)
    return oneThird, twoThird


def buildQuery(df, factorColumns):
    print("\n*************************")
    oneThird, twoThirds = getPercentileValues(df)

    # Build select clause.
    select = "SELECT "
    counter = 0
    for colName in factorColumns:
        select += " AVG(" + colName + ")"
        if (counter < len(factorColumns) - 1):
            select += ","
        counter += 1
    select += " FROM Marketing "

    # Query top satisfaction summary
    sql = select + " WHERE MntWines > " + str(twoThirds)
    results1 = executeSQL(sql, df)

    # Query middle satisfaction summary
    sql = select + " WHERE MntWines >= " + str(oneThird) + \
          " AND MntWines <= " + str(twoThirds)
    results2 = executeSQL(sql, df)

    # Query low satisfaction summary
    sql = select + " WHERE MntWines < " + str(oneThird)
    results3 = executeSQL(sql, df)

    summaryDf = results1.copy()
    summaryDf = summaryDf.append(results2)
    summaryDf = summaryDf.append(results3)
    print(summaryDf)
    return results1, results2, results3


factorColumns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
                 'Response']

COLORS = ['red', 'green', 'orange', 'yellow',
          'purple', 'blue', 'pink', 'brown',
          'black', 'cyan', 'gray']

subDf = data[factorColumns]
print(data['MntWines'])
high, medium, low = buildQuery(subDf, factorColumns)

# Show attribute plots for different satisfaction ranges.
import matplotlib.pyplot as plt

plt.subplots(nrows=1, ncols=3, figsize=(14, 7))

# High satisfaction.
plt.subplot(1, 3, 1)
highTransposed = high.T.reset_index()
plt.bar(highTransposed['index'], highTransposed[0], color=COLORS)
plt.title("High wine purchase")
plt.ylim(0, 9)
plt.xticks(rotation=70)

# Medium satisfaction.
plt.subplot(1, 3, 2)
medTransposed = medium.T.reset_index()
plt.bar(medTransposed['index'], medTransposed[0], color=COLORS)
plt.title("Medium wine purchse")
plt.ylim(0, 9)
plt.xticks(rotation=70)

# Low satisfaction.
plt.subplot(1, 3, 3)
lowTransposed = low.T.reset_index()
plt.bar(lowTransposed['index'], lowTransposed[0], color=COLORS)
plt.title("Low wine purchase")
plt.ylim(0, 9)
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()
