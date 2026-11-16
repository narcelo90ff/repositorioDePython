"""2. Bibliotecas e preparação do arquivo

Importa as bibliotecas necessárias, define parâmetros de reprodução e verifica se o arquivo CSV está disponível."""

from pathlib import import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
CSV_NAME = 'Wholesale_customers_data.csv'

if not Path(CSV_NAME).exists():
    try:
        from google.colab import files
        print('Envie o arquivo Wholesale_customers_data.csv')
        files.upload()
    except Exception:
        pass

if not Path(CSV_NAME).exists():
    raise FileNotFoundError('Coloque Wholesale_customers_data.csv na pasta do notebook ou faça upload no Colab.')

"""3. Leitura e exploração inicial

Carrega a base, seleciona as seis variáveis de gastos e verifica a estrutura e a qualidade dos dados. Channel e Region são consultadas para interpretação, mas não entram no treinamento."""

df = pd.read_csv(CSV_NAME)
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
spending = df[features].astype(float)

print('Dimensão da base (linhas, colunas):', df.shape)
print('Quantidade de clientes:', len(df))
print('Valores ausentes na base:', int(df.isna().sum().sum()))
print('Linhas duplicadas:', int(df.duplicated().sum()))
print('Distribuição de Channel (somente consulta):')
print(df['Channel'].value_counts().sort_index())
print('Distribuição de Region (somente consulta):')
print(df['Region'].value_counts().sort_index())

assert len(df) == 440, 'A base esperada tem 440 linhas. Confira o arquivo utilizado.'
assert spending.isna().sum().sum() == 0, 'Há valores ausentes nas categorias de gasto.'

"""4. Pré-processamento: logaritmo, divisão e padronização

Aplica log1p aos gastos, separa os dados em treinamento e teste e padroniza as variáveis usando os parâmetros aprendidos no treinamento."""

X_log = np.log1p(spending)
X_train_raw, X_test_raw = train_test_split(X_log, test_size=0.20, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print('Formato do treino:', X_train.shape)
print('Formato do teste:', X_test.shape)

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot([X_log[c] for c in features])
ax.set_xticklabels(features)
ax.set_title('Distribuição dos gastos após log1p')
ax.set_ylabel('log(1 + gasto anual)')
ax.tick_params(axis='x', rotation=25)
ax.grid(axis='y', alpha=.3)
plt.tight_layout()
plt.show()

"""Assimetria e correlações

Compara a assimetria e as correlações antes e depois da transformação log1p, exibindo os resultados em uma tabela e em um mapa de calor."""

print('Assimetria antes e depois do log1p:')
display(pd.DataFrame({'bruto': spending.skew(), 'log1p': X_log.skew()}).round(2))
corr_raw = spending.corr()
corr_log = X_log.corr()
print('Milk-Grocery: bruta =', round(corr_raw.loc['Milk', 'Grocery'], 2), '| após log1p =', round(corr_log.loc['Milk', 'Grocery'], 2))
print('Grocery-Detergents_Paper: bruta =', round(corr_raw.loc['Grocery', 'Detergents_Paper'], 2), '| após log1p =', round(corr_log.loc['Grocery', 'Detergents_Paper'], 2))

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr_log.values, cmap='YlGnBu', vmin=-1, vmax=1)
ax.set_xticks(range(6), features, rotation=35, ha='right')
ax.set_yticks(range(6), features)
for a in range(6):
    for b in range(6):
        ax.text(b, a, f'{corr_log.values[a, b]:.2f}', ha='center', va='center', fontsize=8)
ax.set_title('Correlação entre categorias de gasto (após log1p)')
fig.colorbar(im, ax=ax, label='Correlação de Pearson')
plt.tight_layout()
plt.show()

"""5. Configurações que serão comparadas

Testa 24 combinações: de 1 a 6 componentes gaussianos e quatro tipos de covariância. As métricas de cada configuração são reunidas para comparação."""

results = []
for covariance in ['full', 'tied', 'diag', 'spherical']:
    for k in range(1, 7):
        gm = GaussianMixture(n_components=k, covariance_type=covariance, random_state=RANDOM_STATE, n_init=10, reg_covar=1e-5, max_iter=1000)
        gm.fit(X_train)
        results.append({
            'n_components': k,
            'covariance_type': covariance,
            'BIC': gm.bic(X_train),
            'AIC': gm.aic(X_train),
            'test_log_likelihood_mean': gm.score(X_test),
            'converged': gm.converged_,
            'iterations': gm.n_iter_
        })

results = pd.DataFrame(results)
results.to_csv('comparacao_modelos.csv', index=False)
print('Melhor BIC:')
display(results.loc[[results['BIC'].idxmin()]])
print('Melhor AIC:')
display(results.loc[[results['AIC'].idxmin()]])
print('Melhor log-verossimilhança média no teste:')
display(results.loc[[results['test_log_likelihood_mean'].idxmax()]])

bic_pivot = results.pivot(index='covariance_type', columns='n_components', values='BIC')
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(bic_pivot.values, aspect='auto', cmap='YlGnBu_r')
ax.set_xticks(range(6), range(1, 7))
ax.set_yticks(range(4), bic_pivot.index)
ax.set_xlabel('Número de componentes')
ax.set_ylabel('Covariância')
ax.set_title('BIC por configuração (menor é melhor)')
fig.colorbar(im, ax=ax, label='BIC')
plt.tight_layout()
plt.show()

"""Curvas de BIC, AIC e log-verossimilhança de teste

Apresenta as curvas das métricas para comparar as configurações. BIC e AIC são calculados no treinamento; a log-verossimilhança média é avaliada no teste."""

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for covariance in ['full', 'tied', 'diag', 'spherical']:
    part = results[results['covariance_type'] == covariance]
    axes[0].plot(part['n_components'], part['BIC'], marker='o', label=covariance)
    axes[1].plot(part['n_components'], part['AIC'], marker='o', label=covariance)
    axes[2].plot(part['n_components'], part['test_log_likelihood_mean'], marker='o', label=covariance)
for ax, title in zip(axes, ['BIC (treino) - menor é melhor', 'AIC (treino) - menor é melhor', 'Log-verossimilhança média no teste - maior é melhor']):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Número de componentes (K)')
    ax.grid(alpha=.3)
axes[0].legend(title='Covariância')
plt.tight_layout()
plt.show()

"""6. Escolha do modelo e ajuste final

Seleciona a configuração com menor BIC e ajusta o modelo escolhido aos 440 clientes para descrever os perfis da base completa."""

best_bic = results.loc[results['BIC'].idxmin()]
selected_k = int(best_bic['n_components'])
selected_cov = str(best_bic['covariance_type'])
print('Configuração selecionada pelo BIC:', selected_k, selected_cov)

scaler_all = StandardScaler()
X_all = scaler_all.fit_transform(X_log)
final_model = GaussianMixture(n_components=selected_k, covariance_type=selected_cov, random_state=RANDOM_STATE, n_init=20, reg_covar=1e-5, max_iter=1000)
final_model.fit(X_all)
labels = final_model.predict(X_all)
probabilities = final_model.predict_proba(X_all)
result_df = df.copy()
result_df['cluster'] = labels
result_df.to_csv('clientes_com_cluster.csv', index=False)

"""7. Métricas, tamanhos dos componentes e probabilidades

Mostra o tamanho e os pesos dos componentes, as métricas do ajuste final e as probabilidades estimadas de pertencimento de cada cliente."""

print('Tamanho dos componentes:')
display(result_df['cluster'].value_counts().sort_index().rename('clientes').to_frame())
print('Pesos da mistura:', np.round(final_model.weights_, 4))
print('BIC no ajuste final:', round(final_model.bic(X_all), 2))
print('AIC no ajuste final:', round(final_model.aic(X_all), 2))
print('Log-verossimilhança média:', round(final_model.score(X_all), 4))

if len(np.unique(labels)) > 1:
    print('Silhouette complementar:', round(silhouette_score(X_all, labels), 4))
else:
    print('Silhouette não calculado: foi encontrado apenas um componente.')

max_probability = probabilities.max(axis=1)
print('Probabilidade máxima média:', round(max_probability.mean() * 100, 1), '%')
print('Clientes com probabilidade máxima < 0,70:', round((max_probability < .70).mean() * 100, 1), '%')

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(max_probability, bins=np.linspace(0, 1, 11), edgecolor='white')
ax.axvline(.70, color='tomato', linestyle='--', label='Referência ilustrativa 0,70')
ax.set_title('Maior probabilidade posterior por cliente')
ax.set_xlabel('Maior probabilidade')
ax.set_ylabel('Número de clientes')
ax.legend()
plt.tight_layout()
plt.show()

ambiguous = np.argsort(max_probability)[:5]
for idx in ambiguous:
    print(f'Cliente {idx + 1}: componente previsto={labels[idx]}, probabilidades={np.round(probabilities[idx], 3)}')

"""8. Interpretação dos perfis de compradores

Resume os gastos por componente e cruza os grupos com Channel e Region para ajudar a interpretar os perfis identificados."""

features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
profile_median = result_df.groupby('cluster')[features].median().round(1)
profile_mean = result_df.groupby('cluster')[features].mean().round(1)

print('Medianas por componente:')
display(profile_median)
print('Médias por componente:')
display(profile_mean)
print('Distribuição posterior por canal:')
display(pd.crosstab(result_df['cluster'], result_df['Channel'], normalize='index').round(3))
print('Distribuição posterior por região:')
display(pd.crosstab(result_df['cluster'], result_df['Region'], normalize='index').round(3))

fig, ax = plt.subplots(figsize=(9, 4))
profile_median.T.plot(kind='bar', ax=ax)
ax.set_title('Mediana de gastos por componente')
ax.set_ylabel('Unidades monetárias (m.u.)')
ax.set_xlabel('Categoria de produto')
ax.legend(title='Componente')
ax.tick_params(axis='x', rotation=35)
plt.tight_layout()
plt.show()
