import pandas as pd
import statsmodels.api as sm

# read in conjoint survey profiles with respondent ranks
df = pd.read_csv('survey_results.csv')

# Show all columns of data frame.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# This performs a weighted regression for ranking with string variables as
# levels.
attributeNames = [ 'Website', 'Purchase', 'Shipping']

y = df[['Rank']]
X = df[attributeNames]
X = pd.get_dummies(X, columns =attributeNames )
X = sm.add_constant(X)

lr_model = sm.OLS(y, X).fit()
print(lr_model.summary())

counter     = 0
levelNames  = list(X.keys())# Level names are taken directly from X column names.
levelNames.pop(0)           # Remove constant for intercept.
ranges      = []

# Store all part-worth (utility) values in a list.
# The values are taken directly from the model coefficients.
utilities   = list(lr_model.params)
utilities.pop(0)            # Removes the intercept value.

# Iterate through all attributes to create part-worths.
for attributeName in attributeNames:
    partWorths                 = []

    # Iterate through all levels.
    for levelName in levelNames:
        # If level name contains the attribute store the part worth.
        if(attributeName in levelName):
            partWorth = utilities[counter] # Store corresponding model coefficient.
            print(" :", levelName + ": " + str(partWorth))
            partWorths.append(partWorth)
            counter += 1

    # Summarize utility range for the attribute.
    partWorthRange = max(partWorths) - min(partWorths)
    ranges.append(partWorthRange)

# Calculate relative importance scores for each attribute.
importances = []
for i in range(0, len(ranges)):
    importance = 100*ranges[i]/sum(ranges)
    importances.append(importance)
    print(attributeNames[i] + " importance: " + str(importance))

import matplotlib.pyplot as plt

# Show the importance of each attribute.
plt.bar(attributeNames, importances)
plt.title("Attribute Importance")
plt.show()

# Show user's preference for all levels.
plt.bar(levelNames, utilities)
plt.title("Level Part-Worths Representing a Person’s Preferences")
plt.xticks(rotation=80)
plt.show()
