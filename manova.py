import pandas as pd
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#VERSIONE 1: 2 COMPONENTI
# df = pd.read_excel('results.xlsx')

# print(df.head())

# X = df.iloc[:, 4:13].values  # variabili di risposta
# y = pd.get_dummies(df['Technique'], drop_first=True).values  

# #Canonical Correlation Analysis (CCA)
# cca = CCA(n_components=2) 
# X_c, y_c = cca.fit_transform(X, y)

# df['Canonical Score 1'] = X_c[:, 0]
# df['Canonical Score 2'] = X_c[:, 1]


# plt.figure(figsize=(10, 6))
# palette = sns.color_palette("tab20", df['Technique'].nunique())  # 20 colori
# sns.scatterplot(x='Canonical Score 1', y='Canonical Score 2', hue='Technique', data=df, palette=palette)
# plt.title('Canonical Scores by Technique')
# plt.xlabel('Canonical Score 1')
# plt.ylabel('Canonical Score 2')
# plt.legend(title='Technique', bbox_to_anchor=(1.05, 1), loc='best')
# plt.grid(True)
# plt.show()


#PIù COMPONENTI CANONICHE E PAIR PLOTS
# df = pd.read_excel('results.xlsx')

# print(df.head())

# X = df.iloc[:, 4:13].values  # variabili di risposta
# y = pd.get_dummies(df['Technique']).values  

# #Canonical Correlation Analysis (CCA)
# cca = CCA(n_components=2) 
# X_c, y_c = cca.fit_transform(X, y)

# # Aggiungi le componenti canoniche al dataframe
# for i in range(X_c.shape[1]):
#     df[f'Canonical Score {i+1}'] = X_c[:, i]

# # Grafico a coppie per tutte le componenti
# sns.pairplot(df, hue='Technique', vars=[f'Canonical Score {i+1}' for i in range(X_c.shape[1])])
# plt.legend(title='Technique', bbox_to_anchor=(1.05, 1), loc='upper center')
# plt.show()

#VERSIONE 3: COEFFICIENTE DI CORRELAZIONE CANONICO
# df = pd.read_excel('results.xlsx')

# print(df.head())

# X = df.iloc[:, 4:13].values  # variabili di risposta
# y = pd.get_dummies(df['Technique'], drop_first=True).values  

# cca = CCA(n_components=2) 
# X_c, y_c = cca.fit_transform(X, y)

# canonical_corr = cca.score(X, y)
# print(f'Canonical Correlations: {canonical_corr}')

#MATRICE DI CORRELAZIONE
df = pd.read_excel('results.xlsx')
print(df.head())

X = df.iloc[:, 4:13].values  # variabili di risposta
y = pd.get_dummies(df['Technique'], drop_first=True).values  


cca = CCA(n_components=2) 
X_c, y_c = cca.fit_transform(X, y)
# Calcola la matrice di correlazione tra variabili di risposta e punteggi canonici
corr_matrix = np.corrcoef(X_c.T, X.T)[:X_c.shape[1], X_c.shape[1]:]

# Visualizza la matrice di correlazione
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=df.columns[4:13], yticklabels=[f'Canonical Score {i+1}' for i in range(X_c.shape[1])])

plt.show()
