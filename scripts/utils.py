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