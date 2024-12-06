from matplotlib import pyplot as plt
import pandas as pd

df_promo = pd.read_csv("Daten/promotional.csv")
df_good = pd.read_csv("Daten/good.csv")


df_promo["good_data"] = 0

df_good["good_data"] = 1
df_good["advert"] = 0
df_good["pr"] = 0
df_good["fanpov"] = 0
df_good["coi"] = 0
df_good["resume"] = 0


both_data = [df_good, df_promo]
df_both = pd.concat(both_data)
df_grouped = df_both.groupby(["good_data","advert", "pr", "fanpov", "coi", "resume"])["good_data"].count()

fig = plt.figure()
df_grouped.plot(kind="barh", title = "Die Verteilung der Artikel aufgeteilt nach Klassifizierung"
             , xlabel = "Anzahl Artikel", ylabel = "Zugeteilte Kategorien (Good Data, Advert, PR, Fanpov, COI, Resume)")


plt.show()