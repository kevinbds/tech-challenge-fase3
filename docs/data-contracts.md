# Dados e critérios de preparação

## Aquisição e versões

As oito fontes do INEP e do IBGE são identificadas em
[`config/sources.lock.json`](../config/sources.lock.json) por URL, período,
tamanho e SHA-256. A camada Bronze preserva o conteúdo recebido.

`acquire_sources(cache_dir, mode, expected_manifest)` oferece dois modos:

- `reproduce`: recupera os arquivos previstos e exige os mesmos hashes;
- `update`: adquire as fontes novamente e registra outra versão quando o conteúdo muda.

O download tenta HTTPS primeiro. O fallback HTTP é limitado ao host público
`download.inep.gov.br` e fica registrado no manifesto da aquisição. A comparação
por hash verifica a identidade do arquivo recuperado com a versão fixada.

## Camadas e unidades de análise

`prepare_data(cache_dir, manifest_path)` valida os arquivos e materializa Silver,
Gold e os conjuntos de modelagem em armazenamento privado. O retorno informa,
entre outros caminhos, `marts_dir`, `coverage_path` e `receipt_path`.

Silver conserva os campos individuais necessários à preparação. Gold reúne
indicadores e comparações municipais. O conjunto individual de modelagem associa
o rótulo oficial do aluno ao contexto municipal da mesma rede no ano anterior.
Identificadores e registros individuais permanecem fora das saídas públicas.

## População elegível

A classificação considera segundo ano do ensino fundamental, redes estadual e
municipal, rótulo oficial binário e peso finito maior que zero. Os códigos de rede
são interpretados conforme o dicionário de cada arquivo: a codificação dos dados
individuais difere da utilizada nos indicadores agregados.

Registros sem município, escola, aluno, rede ou UF são separados antes das demais
exclusões. Em 2025, essa regra atingiu 628 dos 2.222.792 registros recebidos, com
município, escola e rede simultaneamente ausentes. As exclusões são contadas por
motivo e UF; identificadores não são imputados. Códigos desconhecidos e relações
incompatíveis entre município e UF interrompem a preparação.

`ID_ESCOLA` é um identificador mascarado, inadequado para cruzamento externo com
o Censo Escolar. Proficiência individual, presença e caderno não entram como
atributos preditivos.

## Indicadores, metas e contexto

| Componente | Regra |
| --- | --- |
| Níveis de alfabetização | As nove proporções são estruturalmente indisponíveis em 2023. Em 2024–2025, exigem valores finitos de 0–100 e soma entre 99,5 e 100,5. |
| Metas | A edição 2024 cobre 2024–2030. `>80` conserva o operador estrito; números simples usam `>=`. Meta ausente é diferente de meta não atingida. |
| Precisão numérica | Taxas e metas são armazenadas em `DECIMAL(38,9)`. A comparação usa o valor completo, antes da formatação para exibição. |
| INSE 2021 | Redes estadual e municipal, localização total. Quantidade zero indica contexto indisponível. |
| Censo 2022 | Agregado 4714, variáveis 93 e 614: população e densidade municipais. |
| Diretório IBGE | Valida município, UF e região. A versão utilizada contém 5.571 municípios; o Censo contém 5.570. |

Municípios novos continuam na população elegível, com contexto histórico ausente
quando necessário. As junções exigem chaves dimensionais únicas e preservam a
quantidade de linhas. O contexto preditivo usa exatamente `ano - 1` na mesma rede
e município; uma lacuna não é substituída por um ano mais antigo.

O produto Gold `evolucao_alfabetizacao` mantém, separadamente, a variação em relação
à última observação disponível. Ele é ordenado por município, rede e ano. Essa
definição descritiva não substitui o contexto anual exigido pelos modelos.

## Cobertura temporal

RR não foi avaliado em 2024, e os resultados estaduais dessa edição não incluem
RR e DF. Os registros individuais e municipais disponíveis do DF são mantidos.
O resultado nacional utilizado vem da planilha de referência de 2024; sua
ausência para 2025 nessa planilha não significa ausência de avaliação naquele ano.

As metas futuras também pertencem à edição de 2024 e podem ser revistas pelo
órgão responsável. O conjunto municipal de 2026 contém contexto observado de
2025 e desfecho ausente. Sua população corresponde aos municípios com resultado
municipal disponível em 2025.

## Integridade e consulta

`preparation-receipt.json` registra hashes e tamanhos dos arquivos preparados;
`coverage.json` reúne as contagens e a cobertura. A preparação deve terminar antes
do consumo dos dados: o marcador `preparation.in-progress` e arquivos `.partial`
indicam trabalho ainda em andamento.

Como os dados foram coletados retrospectivamente, suas revisões não representam
necessariamente os arquivos publicados em cada data histórica de previsão.
Diferenças de ordenação física também podem alterar o hash de um Parquet sem
alterar seu conteúdo analítico. O recibo de cada execução permanece associado aos
arquivos efetivamente utilizados naquela execução.
