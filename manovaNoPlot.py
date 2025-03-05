import pandas as pd
from statsmodels.multivariate.manova import MANOVA

df = pd.read_excel('results.xlsx')

print(df.head())

response_vars = '+'.join(df.columns[4:13]) 
formula = f'{response_vars} ~ Technique'

maov = MANOVA.from_formula(formula, data=df)
print(maov.mv_test())