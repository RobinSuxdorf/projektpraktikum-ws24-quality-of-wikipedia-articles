"""Graph the distribution of the labels in the promotional dataset

Author: Emmanuelle Steenhof
"""

from matplotlib import pyplot as plt
import pandas as pd

promo_path = "data/raw/promotional.csv"
good_path = "data/raw/good.csv"

df_promo = pd.read_csv(promo_path)
# df_good = pd.read_csv(good_path)

concatenated_labels = []
for i in range(len(df_promo)):
    y = ""
    x = df_promo.iloc[i]
    if x["advert"] == 1:
        y = y + "advert,"
    if x["pr"] == 1:
        y = y + "pr,"
    if x["fanpov"] == 1:
        y = y + "fanpov,"
    if x["coi"] == 1:
        y = y + "coi,"
    if x["resume"] == 1:
        y = y + "resume,"
    y = y[: len(y) - 1]
    concatenated_labels.append(y)

df_promo["categories"] = concatenated_labels

df_aggreg = df_promo.groupby("categories")["categories"].count()
df_aggreg.plot(
    kind="barh",
    title="Verteilung der Kategorien",
    ylabel="Kategorien",
    xlabel="Anzahl Artikel pro Klasse",
)
plt.show()
