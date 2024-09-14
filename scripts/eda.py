#all eda methods 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

def descriptive_statistics(df,columns):
    for col in columns:
        print("##############",col)
        # Calculate descriptive statistics 
        descriptive_stats = df[col].describe()

        
        # Print the results
        print("Descriptive Statistics:\n", descriptive_stats)


def show_histograms(df, columns):
    for col in columns:
        plt.figure(figsize=(10, 5))
        plt.hist(df[col], bins=30)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        # plt.savefig(f"screenshots/histogram_{col}.png")  
        plt.show()

def show_bar_chart(df, columns):
    for col in columns:
        plt.figure(figsize=(10, 5))
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Bar Chart of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        # plt.savefig(f"screenshots/bar_chart_{col}.png")  
        plt.show()

# Create a scatter plot with color-coded points by postal code
def corr_trotalpremium_totalclaim_postalcode(df):
    sns.scatterplot(x='TotalPremium_Change', y='TotalClaims_Change', data=df, hue='PostalCode',palette='viridis')

    # Set the title and labels
    plt.title('Scatter Plot of TotalPremium Changes vs. TotalClaims Changes')
    plt.xlabel('TotalPremium Changes')
    plt.ylabel('TotalClaims Changes')
    plt.savefig('screenshots/scatter_plot_totalpremium_changes_va_totalclaims_changes.png')  


    # Show the plot
    plt.show()


def analyze_geographic_trends(df):
    # Ensure the TransactionMonth column is in datetime format
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], format='%Y-%m')

    # Sort the data by PostalCode and TransactionMonth
    df = df.sort_values(by=['PostalCode', 'TransactionMonth'])

    # Group by PostalCode and TransactionMonth, and aggregate the data
    trends = df.groupby(['PostalCode', 'TransactionMonth']).agg({
        'CoverType': 'count',            
        'TotalPremium': 'mean',          
        'make': lambda x: x.value_counts().idxmax()  # Most common auto make by postal code and month
    }).reset_index()

    # Calculate the monthly percentage change for the aggregated fields
    trends['TotalPremium_Change'] = trends.groupby('PostalCode')['TotalPremium'].pct_change()

    # Visualize trends over time for TotalPremium, by postal code
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='TransactionMonth', y='TotalPremium', hue='PostalCode', data=trends, palette='tab10')
    plt.title('Trend of TotalPremium Over Time by PostalCode')
    plt.xlabel('Month')
    plt.ylabel('Average TotalPremium')
    plt.xticks(rotation=45)
    plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Visualize trends for CoverType counts over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='TransactionMonth', y='CoverType', hue='PostalCode', data=trends, palette='Set2')
    plt.title('Trend of CoverType Counts Over Time by PostalCode')
    plt.xlabel('Month')
    plt.ylabel('Number of CoverType')
    plt.xticks(rotation=45)
    plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Visualize the most common car make by postal code
    plt.figure(figsize=(12, 6))
    sns.countplot(x='make', hue='PostalCode', data=trends, palette='coolwarm')
    plt.title('Most Common Car Make by PostalCode')
    plt.xticks(rotation=45)
    plt.xlabel('Car Make')
    plt.ylabel('Count')
    plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def outlier_detection(df, columns):

    for col in columns:

        sns.boxplot(x=col, data=df)
        plt.title(f'Box Plot of {col}')
        plt.savefig(f'screenshots/outlier_for_{col}.png')
        plt.show()

def visualize_plots(df):
    # Histogram with density plot overlay for TotalPremium
    sns.histplot(data=df, x='TotalPremium', kde=True, color='skyblue')
    plt.title('Distribution of TotalPremium')
    plt.show()

    # Bar plot with error bars for CoverType vs. TotalPremium
    sns.barplot(x='CoverType', y='TotalPremium', data=df, palette='viridis', ci='sd')
    plt.title('CoverType vs. TotalPremium')
    plt.show()

    # Scatter plot with regression line for TotalPremium vs. TotalClaims
    sns.regplot(x='TotalPremium', y='TotalClaims', data=df, color='orange')
    plt.title('Relationship between TotalPremium and TotalClaims')
    plt.show()