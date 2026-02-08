import matplotlib.pyplot as plt
import seaborn as sns


def plot_label_distribution(series, title="Label distribution"):
    ax = sns.countplot(x=series)
    ax.set_title(title)
    return ax
