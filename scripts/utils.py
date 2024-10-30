import seaborn as sns
import matplotlib.pyplot as plt
def plot_histograms(fraud_df, numerical_features):
    """Plots histograms with KDE for numerical features."""
    plt.figure(figsize=(15, 4))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(1, len(numerical_features), i)
        sns.histplot(fraud_df[feature], bins=30, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def visualize_bivariate_relationship(data, group_by_column):
    """Plots scatter plots for purchase_value vs age, colored by a specified category."""
    plt.figure(figsize=(8, 4))
    
    # Use the group_by_column for coloring the scatter plot
    sns.scatterplot(data=data, x='age', y='purchase_value')  
    
    # Change the title as needed
    plt.title('Purchase Value vs Age')  
    plt.xlabel('Age')
    plt.ylabel('Purchase Value')
    
    plt.legend(title=group_by_column)  # Add legend for the category
    plt.show()