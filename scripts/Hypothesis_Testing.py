# A/B Hypothesis Testing methods

import pandas as pd
from scipy.stats import chi2_contingency


def AB_haypothe(df,col):
    # Calculate loss ratio
    df["LossRatio"] = df["TotalClaims"] / df["SumInsured"]

    # Define high-risk threshold
    high_risk_threshold = 1

    # Create high-risk indicator
    df["HighRisk"] = df["LossRatio"] > high_risk_threshold

    # Create contingency table
    contingency_table = pd.crosstab(df[col], df["HighRisk"])

    # Perform chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return chi2, p_value, dof, expected


def calculate_profit_margin_and_anova(df):
   

    # Calculate profit margin
    df['ProfitMargin'] = (df['TotalPremiums'] - df['TotalClaims']) / df['TotalPremiums']

    # Group df by zip code and calculate average profit margin
    grouped_df = df.groupby("PostalCode")["ProfitMargin"].mean()

    # Perform ANOVA
    f_statistic, p_value = f_oneway(*grouped_df)

    return f_statistic, p_value




def calculate_loss_ratio_and_t_test_for_gender(data, gender_column, loss_ratio_column):
    
    # Calculate loss ratio if it doesn't exist
    if loss_ratio_column not in data.columns:
        data[loss_ratio_column] = data['TotalClaims'] / data['SumInsured']

    # Group data by gender
    female_data = data[data[gender_column] == "Female"][loss_ratio_column]
    male_data = data[data[gender_column] == "Male"][loss_ratio_column]

    # Perform t-test
    t_statistic, p_value = ttest_ind(female_data, male_data)

    return t_statistic, p_value





