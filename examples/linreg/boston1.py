import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("BostonHousing.csv")

# fig = sns.pairplot(df[["rm", "medv"]])
# fig.savefig('pairplot_rm_medv.png')
#
# fig = sns.pairplot(df)
# fig.savefig('pairplot_all.png')

ax = sns.regplot("rm", "medv", data=df[["rm", "medv"]])
fig = ax.get_figure()
fig.savefig('regplot_rm_medv.png')
