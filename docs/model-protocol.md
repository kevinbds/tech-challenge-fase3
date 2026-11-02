# Protocolo de modelagem

## Tarefas e atributos

A classificação estima risco de não alfabetização: a classe positiva é
`risco = 1 - IN_ALFABETIZADO`. O treinamento preserva as linhas elegíveis e seus
pesos oficiais. A regressão estima a taxa de alfabetização da rede municipal,
com uma observação por município e ano.

Os oito atributos são taxa de alfabetização e média de Português do ano anterior,
INSE municipal de 2021, logaritmo da população e densidade de 2022, UF, região e
rede. Os conjuntos de modelagem registram `ano` e `ano_contexto`; valores de taxa
e Português disponíveis precisam corresponder exatamente ao ano anterior.

Identificadores, proficiência individual, presença, respostas e resultados
contemporâneos ficam fora do conjunto preditivo. As nove proporções de níveis
também são excluídas, pois não estão disponíveis em todos os anos utilizados.

## Divisão dos dados

O desenvolvimento utiliza 2024 com contexto de 2023. A avaliação temporal utiliza
2025 com contexto de 2024. Uma divisão determinística, com semente 42, reserva
20% dos municípios de 2024; a validação cruzada usa cinco divisões agrupadas nos
80% restantes. As tarefas individual e municipal compartilham essa partição.

Imputação numérica, tratamento de categorias, one-hot e escala são ajustados
dentro de cada divisão de treinamento. A mediana e a escala são não ponderadas;
os pesos oficiais entram nos objetivos e nas métricas da classificação.

## Seleção dos modelos

| Tarefa | Famílias | Métrica de seleção | Referências |
| --- | --- | --- | --- |
| Classificação | Regressão logística, Random Forest e HistGradientBoosting | Média da AP ponderada nos cinco folds | Prevalência ponderada e risco contextual anterior |
| Regressão municipal | Ridge, Random Forest e HistGradientBoosting | Média do MAE, com peso igual por município | Persistência da taxa anterior |

A busca avalia quatro configurações lineares e até 12 configurações para cada
família de árvores. Contexto ausente no baseline usa prevalência ou média calculada no
treinamento, conforme a tarefa. As comparações mantêm as mesmas linhas e pesos.

`class_weight=None`, `early_stopping=False` e `oob_score=False` evitam objetivos ou
partições implícitas diferentes da divisão definida. Random Forest treina em
paralelo e usa `n_jobs=1` na inferência, estabilizando a soma das árvores e as
decisões em probabilidades próximas ao limiar.

## Calibração e avaliação

O classificador selecionado é congelado e recebe calibração sigmoid na reserva
de 2024. Os pesos dessa etapa ajustam somente a calibração, pois a base já foi
treinada com seus pesos. O limiar maximiza F2 ponderado; empates favorecem maior
recall e, depois, menor limiar. A avaliação também informa resultados no corte 0,5.

A reserva municipal fornece os quantis 5% e 95% dos resíduos para uma banda
exploratória. Predições da regressão são limitadas a 0–100. A seleção, o limiar e
os quantis são registrados antes da leitura dos rótulos de 2025 pela modelagem.

O teste informa AP, ROC AUC, Brier, precisão, recall, F2 e matrizes de confusão
ponderadas e por contagem. A análise de subgrupos considera UF, região e presença
de contexto anterior. Na regressão, são calculados MAE, RMSE, R², concordância com
metas e cobertura observada das bandas.

Os intervalos são percentis de 500 reamostragens pareadas por município. Eles
descrevem variabilidade condicional aos dados observados e não reproduzem todo
o desenho amostral oficial nem a incerteza de choques futuros.

## Interpretação

A importância por permutação troca atributos entre blocos município/rede e mede
a queda de AP ponderada da probabilidade calibrada. A agregação de entradas
iguais por contexto e classe conserva sua massa de pesos.

SHAP descreve o estimador-base em até 256 contextos distintos de desenvolvimento,
com peso igual entre contextos e indicação da classe e escala explicadas. Sua
interpretação não equivale automaticamente à da probabilidade calibrada.

K-Means resume os perfis municipais observados em 2025, após o encerramento da
seleção preditiva. O número de grupos é escolhido por silhouette no conjunto
municipal completo, e a estabilidade é descrita por ARI. A distribuição por UF e
região serve à leitura territorial dos grupos, sem atribuir causalidade.

Uma experiência separada demonstra vazamento usando a mesma regressão logística
e divisão de 2024, com e sem inclusão indevida do próprio alvo. Esse atributo é
restrito à demonstração e não participa dos modelos avaliados.

## Projeção de 2026 e cenários

Os modelos avaliados em 2025 são mantidos separadamente dos reajustes. O novo
classificador usa desenvolvimento de 2024 e dados de 2025 fora dos municípios
reservados; a reserva de 2025 fornece nova calibração e novo limiar. O regressor
é reajustado em 2024–2025 com os parâmetros selecionados anteriormente.

A projeção de 2026 exige contexto de 2025. Os cenários de 2027–2030 mantêm a taxa
projetada de 2026 e a comparam às metas anuais, preservando operadores e ausências.
Eles não são novas previsões desses anos. Nenhuma métrica independente de 2026
é atribuída aos modelos reajustados.

## Artefatos e resultados observados

Os identificadores e hashes dos resultados correspondem à execução original,
cujo núcleo científico foi preservado. A configuração atual contém apenas
parâmetros científicos, com os mesmos valores consumidos pelos modelos. Seu hash
difere do registrado nas métricas; uma nova execução recebe outro identificador.
O arquivo [`source_version.json`](../reports/source_version.json) relaciona as
versões e os hashes do código científico.

Os arquivos de modelo contêm estimador e metadados. Dados individuais, vetores
de pesos do treinamento e bases de explicação não são exportados. A restauração
confere probabilidades e decisões no limiar específico de cada classificador.

No teste de 2025, Random Forest não demonstrou ganho de AP sobre o contexto
anterior. Seu limiar F2 sinalizou 93,56% dos registros, apesar do recall de 98,03%,
limitando a seletividade. A regressão municipal reduziu o MAE de 12,99 para
11,58 p.p., mas a banda exploratória cobriu 77,33%, abaixo da referência de 90%.
Os valores completos estão em [`reports/metrics.json`](../reports/metrics.json).
