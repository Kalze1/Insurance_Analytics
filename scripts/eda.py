#all eda methods 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

def descriptive_statistics(df, columns):
    # Calculates descriptive statistics for specified columns in a DataFrame.
    results = {}
    for col in columns:
        descriptive_stats = df[col].describe()
        results[col] = descriptive_stats.to_dict()

    return results


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



def visualize_descriptive_statistics(stats):
    # Visualizes descriptive statistics for each column in a dictionary.
    for col, stats_dict in stats.items():
        # Extract relevant statistics
        count = stats_dict['count']
        mean = stats_dict['mean']
        std = stats_dict['std']
        min_value = stats_dict['min']
        q25 = stats_dict['25%']
        q50 = stats_dict['50%']
        q75 = stats_dict['75%']
        max_value = stats_dict['max']

        # Create a figure and axes
        fig, ax = plt.subplots()

        # Plot the statistics as a bar chart
        bars = ax.bar([1, 2, 3, 4, 5, 6, 7, 8], [count, mean, std, min_value, q25, q50, q75, max_value])

        # Set labels and title
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
        ax.set_xticklabels(['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])
        ax.set_ylabel(col)
        ax.set_title(f'Descriptive Statistics for {col}')

        # Add labels to the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), ha='center', va='bottom')

        # Show the plot
        # plt.savefig(f'screenshots/discriptive_{col}')
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