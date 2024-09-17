# A/B Hypothesis Testing methods

import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, ttest_ind

from scripts.preprocessing import replace_missing_with_mean


def AB_haypothe(df,col):
    # Calculate loss ratio
    df["LossRatio"] = df["TotalClaims"] / df["SumInsured"]

    # Define high-risk threshold
    high_risk_threshold = 128.8991

    # Create high-risk indicator
    df["HighRisk"] = df["LossRatio"] > high_risk_threshold

    # Create contingency table
    contingency_table = pd.crosstab(df[col], df["HighRisk"])

    # Perform chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return chi2, p_value, dof, expected




def calculate_profit_margin_and_anova(df):
    # Calculate profit margin
    df['ProfitMargin'] = (df['TotalPremium'] - df['TotalClaims']) / df['TotalPremium']
    df['LossRatio'] = df['TotalClaims'] / df['SumInsured']
    
    # Replace missing values in ProfitMargin and LossRatio columns
    df = replace_missing_with_mean(df, ['ProfitMargin', 'LossRatio'])

    # Group df by zip code and calculate average profit margin per group
    grouped_df = df.groupby("PostalCode")["ProfitMargin"].apply(list)

    # Remove groups with less than two values (ANOVA requires at least two groups with multiple observations)
    grouped_df = grouped_df[grouped_df.apply(lambda x: len(x) > 1)]

    # Perform ANOVA if there are at least two groups
    if len(grouped_df) > 1:
        f_statistic, p_value = f_oneway(*grouped_df)
    else:
        f_statistic, p_value = None, None  # Not enough groups for ANOVA

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





