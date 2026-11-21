# Model Card — Previsão de Tempo Total de Entrega (Meituan ETA)

> **Status:** experimental / benchmark. **Não aprovado para uso em produção.**
> **Versão do card:** 1.0 — 2026-11-21.
> **Fonte de verdade dos números:** MLflow local em `http://localhost:5001`, experimento
> `meituan-eta` (`experiment_id=1`). Este documento é uma fotografia das execuções
> registradas; se houver divergência, o MLflow prevalece.

Este Model Card cobre as **seis configurações de modelo** produzidas na Etapa 3
(três algoritmos × dois estágios de features) e a execução de referência
`baseline-meituan`. Ele foi escrito para atender aos itens pedidos na Etapa 4:
objetivo, dados, população/contexto, métricas, limitações, usos recomendados, usos não
recomendados, riscos, aspectos éticos e responsável pelo modelo.

---

## Resumo executivo

| Pergunta | Resposta curta |
| --- | --- |
| O que o modelo faz? | Estima, em minutos, o tempo total entre a criação de um pedido de delivery e a chegada ao cliente. |
| Qual é o melhor resultado? | `hist_gradient_boosting-complete`: MAE de teste **5,464 min**, RMSE **7,040 min**, R² **0,641**; MAE **42,5%** menor que o MAE da estimativa oficial da Meituan no mesmo split. |
| Qual configuração usar em cada momento? | `hist_gradient_boosting-reduced` no checkout (antes de haver entregador) e `hist_gradient_boosting-complete` no despacho/aceite. |
| Qual é a limitação mais séria? | O `poi_target_enc` foi calculado fora da dobra, mas **antes** do split train/test; por isso as métricas de teste são otimistas. |
| Pode definir a promessa ao cliente? | **Não diretamente.** O modelo é uma previsão pontual e não calibrada para nível de serviço; uma promessa exige margem/quantil superior e backtest. |
| Pode avaliar entregadores? | **Não.** O alvo mede o sistema de entrega, não o desempenho individual; há confundimento por distrito, loja, clima e oferta. |
| Onde consultar? | MLflow em `http://localhost:5001/#/experiments/1`; os run IDs estão na seção 2. |
| Há monitoramento em produção? | **Não implementado.** A seção 13 propõe indicadores e gatilhos para a Etapa 4. |

---

## 1. Objetivo

### 1.1 Pergunta de negócio

Quanto tempo, em minutos, um pedido de delivery **imediato** leva entre a criação
(`platform_order_time`) e a chegada ao cliente (`arrive_time`)?

A previsão serve para apoiar decisões operacionais em dois momentos distintos:

| Momento | Estágio | O que se sabe | Para que serve |
| --- | --- | --- | --- |
| Checkout / compra | `reduced` | Restaurante, cliente, distância, horário, distrito, previsão de preparo e encoding do POI. Ainda não há entregador. | Estimar o tempo antes de existir alocação; apoiar a comunicação inicial e o planejamento. |
| Despacho / aceite | `complete` | Tudo do checkout + posição do entregador, tempo até o aceite, fila, carga, recusas, demanda do distrito e wave. | Atualizar a estimativa depois que o entregador aceitou e há mais informação operacional. |

### 1.2 Definição formal do alvo

```text
target_total_time_min = (arrive_time - platform_order_time) / 60
```

- Unidade: minutos.
- Faixa usada no treino: **5 a 90 min** (filtro de coerência da Etapa 3).
- Alvo é a duração **realizada**, não a promessa ao cliente.
- A referência oficial (`meituan_baseline_min`) é:

```text
meituan_baseline_min = (estimate_arrived_time - platform_order_time) / 60
```

`estimate_arrived_time` é a **promessa** exibida ao consumidor, com margem de segurança
deliberada; não é uma previsão pontual do tempo realizado. Essa diferença é central para
interpretar a comparação da seção 5.

### 1.3 O que o modelo não é

- Não é um modelo de risco de atraso nem de probabilidade de violar a promessa.
- Não é um modelo causal de despacho: prever duração não responde “o que aconteceria se
  este pedido fosse atribuído a outro entregador?”.
- Não é um modelo de precificação, crédito, seguro ou avaliação de pessoas.
- Não é um sistema de produção: não há serving, latência medida, fallback ou monitoramento
  contínuo nesta entrega.

### 1.4 Critério de sucesso

Para a Etapa 3/4, o sucesso não é “ganhar acurácia”. É demonstrar que o resultado é
**compreensível, reproduzível e governável**. Os critérios usados foram:

1. MAE de teste menor que a referência oficial no mesmo split.
2. RMSE e R² registrados para avaliar erros grandes e variância explicada.
3. Pipeline reproduzível por DVC, com versões de dados, código e parâmetros.
4. Cada execução rastreável no MLflow por `run_id`, `data_version`, parâmetros e artefatos.
5. Limitações conhecidas documentadas em vez de escondidas.

---

## 2. Modelos cobertos

### 2.1 Por que seis configurações?

O repositório reproduz o notebook de origem com **três algoritmos tradicionais** e **dois
conjuntos de features**:

- **Random Forest** (`RandomForestRegressor`): ensemble de árvores, robusto e com
  `feature_importances_` nativo; artefatos grandes.
- **Histogram Gradient Boosting** (`HistGradientBoostingRegressor`): boosting rápido,
  bom desempenho e artefato compacto; não expõe `feature_importances_` nativo.
- **LightGBM** (`LGBMRegressor`): boosting eficiente, artefato compacto e
  `feature_importances_` nativo.

Os dois estágios não são “versões” do mesmo modelo para a mesma decisão. Eles respondem à
mesma pergunta em **momentos operacionais diferentes**, com informação disponível diferente:

- `reduced` = 9 features disponíveis no checkout.
- `complete` = 19 features disponíveis após o aceite do entregador.

Por isso, o `complete` não substitui automaticamente o `reduced`: no checkout, as features
do `complete` ainda não existem.

### 2.2 Identificação das execuções

As seis execuções de modelo usam `seed=42`, `n_train=350.000`, `n_val=100.000`; os seis
modelos foram treinados sobre `data/processed/train.parquet` com
`data_version=d1b189634113` (prefixo de SHA-256). O baseline usa o split de teste
(`data_version=127078ad1d3e`).

| Run (MLflow) | `run_id` | Algoritmo | Estágio | Features | Artefato em `models/` | Tamanho | Início (UTC) |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| `random_forest-reduced` | `350847ac20ce4e4284db7a7de754392f` | Random Forest | `reduced` | 9 | `random_forest-reduced.pkl` | 176,2 MB | 2026-11-01T15:57:08.920Z |
| `random_forest-complete` | `9c0f865b5d9b4d98ba46d4951f43f61a` | Random Forest | `complete` | 19 | `random_forest-complete.pkl` | 329,6 MB | 2026-11-01T16:00:23.472Z |
| `hist_gradient_boosting-reduced` | `dea2a29db5194003befa1a5b6297a4ad` | Histogram Gradient Boosting | `reduced` | 9 | `hist_gradient_boosting-reduced.pkl` | 1,8 MB | 2026-11-01T16:05:29.288Z |
| `hist_gradient_boosting-complete` | `2330a52718d0401db1e105454d23470c` | Histogram Gradient Boosting | `complete` | 19 | `hist_gradient_boosting-complete.pkl` | 1,8 MB | 2026-11-01T16:07:36.140Z |
| `lightgbm-reduced` | `a4476ef3a19641de8c7edb38a9cb8f7e` | LightGBM | `reduced` | 9 | `lightgbm-reduced.pkl` | 0,9 MB | 2026-11-01T16:08:57.870Z |
| `lightgbm-complete` | `829dd9d5c959485ca01ee55c4c5c01f2` | LightGBM | `complete` | 19 | `lightgbm-complete.pkl` | 1,0 MB | 2026-11-01T16:11:50.366Z |
| `baseline-meituan` | `cbbac0cad5b24a27bd431008520b74f1` | Referência oficial (sem modelo treinado) | `reference` | — | — | — | 2026-11-01T19:58:26.356Z |

Os tamanhos são os bytes dos pickles em `models/` (1 MB = 10⁶ bytes). O baseline não treina
modelo: ele apenas calcula as quatro métricas a partir da coluna
`estimate_arrived_time`.

### 2.3 Hiperparâmetros registrados

| Run | Hiperparâmetros (valores registrados no MLflow) |
| --- | --- |
| `random_forest-reduced` | `n_estimators=100`, `max_depth=16`, `min_samples_split=10`, `random_state=42`, `n_jobs=-1` |
| `random_forest-complete` | `n_estimators=100`, `max_depth=18`, `min_samples_split=8`, `random_state=42`, `n_jobs=-1` |
| `hist_gradient_boosting-reduced` | `max_iter=500`, `learning_rate=0.08`, `max_leaf_nodes=31`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=20`, `random_state=42` |
| `hist_gradient_boosting-complete` | `max_iter=500`, `learning_rate=0.08`, `max_leaf_nodes=31`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=20`, `random_state=42` |
| `lightgbm-reduced` | `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`, `n_jobs=-1` |
| `lightgbm-complete` | `n_estimators=350`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`, `n_jobs=-1` |

Além desses, cada run registra `model`, `stage`, `seed=42`, `n_train`, `n_val`,
`n_features`, a lista ordenada `features` e `data_sha256`. Os hiperparâmetros vêm de
[`params.yaml`](../../params.yaml), não estão hardcoded no código de treino.

### 2.4 Seleção recomendada

| Decisão | Escolha primária | Justificativa | Alternativa |
| --- | --- | --- | --- |
| Checkout (`reduced`) | `hist_gradient_boosting-reduced` | Melhor MAE do estágio (5,941), melhor R² (0,576), artefato de 1,8 MB. | `lightgbm-reduced` (MAE 5,976; 0,9 MB; importância nativa). |
| Despacho (`complete`) | `hist_gradient_boosting-complete` | Melhor MAE (5,464), melhor RMSE (7,040), melhor R² (0,641), artefato de 1,8 MB. | `lightgbm-complete` (MAE 5,486; diferença de 0,022 min; 1,0 MB; importância nativa). |
| Benchmark/interpretabilidade | Random Forest `complete` | `feature_importances_` nativo e comportamento conhecido. | LightGBM `complete` (mais compacto e quase tão preciso). |

**Ressalvas importantes:**

- As diferenças entre HGB e LightGBM são pequenas (0,022 min no `complete`) e **não foram
  testadas estatisticamente**; a escolha se baseia em uma única split de teste.
- O Random Forest é cerca de 100 a 180 vezes maior que o HGB e ficou atrás em MAE/R²;
  não é a melhor escolha para serving, embora seja útil como benchmark.
- O HGB não tem importância de features nativa. Se explicabilidade por feature for
  requisito, use LightGBM (importância nativa) ou permutation importance/SHAP.
- Nenhuma dessas escolhas constitui aprovação para produção.

---

## 3. Dados utilizados

### 3.1 Origem, período e licença

| Item | Descrição |
| --- | --- |
| Fonte | Dataset do *INFORMS TSL Data-Driven Research Challenge* (Meituan), distribuição do desafio. |
| Arquivos | `all_waybill_info_meituan_0322.csv` (waybills), `courier_wave_info_meituan.csv` (waves), `dispatch_waybill_meituan.csv` e `dispatch_rider_meituan.csv` (entradas de despacho). |
| Período | `dt` de **2022-10-17 a 2022-10-24** (8 dias), horário local UTC+8. |
| Volume bruto | **654.343 waybills**, 23 colunas; 206.748 waves; 15.921 linhas de dispatch waybill; 62.044 linhas de dispatch rider. |
| Unidade | **waybill** (uma tentativa de entrega). Um pedido pode ter mais de um waybill quando é recusado e reentra no despacho. |
| Privacidade | IDs de pedido, waybill, entregador, POI e distrito são anonimizados; coordenadas foram deslocadas uniformemente e não escaladas. |
| Licença | **CC BY-NC 4.0** — uso não comercial; redistribuição dos dados brutos não é permitida. |
| Citação | “This research was supported by data provided by Meituan.” |

Os dados brutos **não são redistribuídos com o repositório**. O remote DVC é um diretório
local fora do repositório, e o pipeline foi desenhado para que os arquivos originais
permaneçam fora do Git. Qualquer uso de dados ou artefatos derivados deve respeitar a
licença CC BY-NC 4.0.

### 3.2 Pipeline e versões

O pipeline da Etapa 3 é executado por DVC em sete estágios:

| Estágio DVC | Módulo | Entrada principal | Saída principal |
| --- | --- | --- | --- |
| `bootstrap` | `meituan_eta.data.bootstrap` | arquivos brutos em `data/` | `data/raw/` |
| `validate` | `meituan_eta.data.validate` | waybill + waves brutos | `data/interim/validated.parquet` |
| `prepare` | `meituan_eta.data.prepare` | `validated.parquet` | `data/interim/clean.parquet` |
| `features` | `meituan_eta.features.build` | `clean.parquet` + waves | `data/processed/features.parquet` |
| `split` | `meituan_eta.data.split` | `features.parquet` | `train/val/test.parquet` |
| `train` | `meituan_eta.train` | `train.parquet` + `val.parquet` | 6 pickles + 6 runs MLflow |
| `evaluate` | `meituan_eta.evaluate` | `test.parquet` + pickles | métricas de teste, baseline e registry |

Versões relevantes:

| Objeto | Identificador | Onde está |
| --- | --- | --- |
| Código de treino | commit Git `0ba89b6c32e76b28ffe65e1e27885e4a67fb0472` | tag `mlflow.source.git.commit` das runs |
| Dados de treino | `data_version=d1b189634113`; SHA-256 completo `d1b18963411331d521c833e58aab1b985881951cc444c2e4a5a134663fc72191` | tag/param `data_sha256` no MLflow |
| Mesmo `train.parquet` no DVC | md5 `787361e823ee9ca17a2b07897b6c1876` | [`dvc.lock`](../../dvc.lock) |
| Dados de teste (baseline) | `data_version=127078ad1d3e`; SHA-256 completo `127078ad1d3eb6375908f1ea5f6b7fb74759fedf705850ac3d084872afe2427e` | tag do baseline |
| Mesmo `test.parquet` no DVC | md5 `ead67b22e79ebe14bd81a08fa61ac4e1` | [`dvc.lock`](../../dvc.lock) |

O `data_version` do MLflow é um **prefixo de 12 caracteres do SHA-256** do arquivo. O md5
do DVC é outro hash, calculado pelo DVC sobre os mesmos bytes. Os dois identificam o mesmo
arquivo, mas **não são comparáveis nem substituíveis** entre si.

### 3.3 Limpeza e população modelada

O funil de preparação removeu 115.908 waybills (17,7% do total):

| Etapa | Linhas restantes | Descarte |
| --- | ---: | ---: |
| Bruto | 654.343 | — |
| `is_courier_grabbed == 1` | 568.546 | 85.797 |
| `is_prebook == 0` | 546.365 | 22.181 |
| Timestamps positivos (sem sentinela 0) | 538.920 | 7.445 |
| Coordenadas positivas | 538.920 | 0 |
| `dispatch_time >= platform_order_time` | 538.920 | 0 |
| `arrive_time > dispatch_time` | 538.920 | 0 |
| Duração entre 5 e 90 min | **538.435** | 485 |

O conjunto limpo tem **538.435 linhas**; o frame de features resultante tem **46 colunas**
e **4.923 POIs**. As waves casaram 100% das linhas. A validação não encontrou `waybill_id` duplicado, violação de domínio
booleano nem violação de cronologia; os principais problemas são sentinelas 0 em
timestamps/coordenadas e a issue conhecida de `wave_start_time` documentada no relatório
de dados do desafio.

### 3.4 Split e amostragem

| Parâmetro | Valor |
| --- | --- |
| Amostra | 500.000 linhas sorteadas de 538.435, `random_state=42` |
| Treino | 350.000 (70%) |
| Validação | 100.000 (20%) |
| Teste | 50.000 (10%) |
| Seed | 42 |
| Estratificação | nenhuma |

O split é aleatório, não temporal. Os três conjuntos são disjuntos, mas o target encoding
foi calculado antes do split (ver seção 6.1).

### 3.5 Features

**Estágio `reduced` (9 features, disponíveis no checkout):**

| Feature | Descrição |
| --- | --- |
| `dist_rest_client_manhattan_km` | Distância L1 restaurante → cliente, em km. |
| `order_hour` | Hora local (UTC+8) da criação do pedido. |
| `day_of_week` | Dia da semana (0 = segunda). |
| `is_weekend` | Indicador de fim de semana. |
| `sin_order_hour`, `cos_order_hour` | Codificação cíclica do minuto do dia. |
| `est_meal_prep_min` | `estimate_meal_prepare_time - platform_order_time`, em minutos. |
| `da_id` | ID anonimizado do distrito comercial; áreas podem se sobrepor geograficamente. |
| `poi_target_enc` | Target encoding out-of-fold e suavizado do POI (5 folds, smoothing 10). |

**Estágio `complete` (19 features = 9 do `reduced` + 10 adicionais, após o aceite):**

| Feature | Descrição |
| --- | --- |
| `dist_driver_rest_manhattan_km` | Distância L1 entregador → restaurante, em km. |
| `dist_total_driver_trip_manhattan_km` | Soma das duas pernas: entregador → restaurante → cliente. |
| `tempo_ate_aceite_min` | `grab_time - platform_order_time`, em minutos. |
| `queue_wait_min` | `dispatch_time - platform_order_time`: tempo até o despacho. |
| `rem_prep_at_dispatch_min` | `estimate_meal_prepare_time - dispatch_time`: preparo restante no despacho. |
| `prior_rejections` | Número de waybills anteriores para o mesmo pedido (recusas). |
| `carga_entregador` | Pedidos concorrentes na bolsa do entregador no instante do aceite. |
| `da_demanda_20min` | Pedidos no mesmo distrito na janela de 20 minutos anteriores. |
| `wave_em_andamento` | Indicador de wave já em andamento no aceite. |
| `pos_na_wave` | Posição do pedido dentro da wave do entregador. |

As features de distância usam coordenadas deslocadas e métrica L1 (Manhattan), não
distância euclidiana nem rota real. `da_id` é tratado como numérico pelo modelo; não há
one-hot encoding explícito para distritos.

### 3.6 Estatísticas descritivas do alvo e da referência

| Estatística | `target_total_time_min` | `meituan_baseline_min` |
| --- | ---: | ---: |
| n | 538.435 | 538.435 |
| Média | 29,458 min | 37,688 min |
| Desvio-padrão | 11,683 min | 9,073 min |
| Mediana | 27,800 min | 35,050 min |
| Mínimo | 5,000 min | 30,000 min |
| Máximo | 89,983 min | 325,033 min |

A promessa oficial é, em média, **8,23 min maior** que o tempo realizado no conjunto
limpo, e tem cauda longa (máximo de 325 min, pois não é truncada em 90 min). Isso é
consistente com a margem de segurança deliberada descrita no notebook de origem.

### 3.7 O que não está nos dados

O modelo **não observa**:

- clima, chuva, temperatura ou qualidade do ar;
- trânsito, obras, acidentes ou velocidade de rota;
- promoções, cupons, picos de demanda ou campanhas;
- oferta de entregadores em tempo real (além de proxies pontuais);
- tempo real de preparo da cozinha (apenas a estimativa);
- características do cliente, do entregador ou do restaurante além de IDs;
- cancelamentos, reembolsos, avaliações ou reclamações;
- dados de outros dias, cidades ou operações.

Essa ausência é uma limitação de validade, não apenas de performance.

---

## 4. População e contexto

### 4.1 Quem está representado

A população modelada é composta por **waybills de pedidos imediatos que foram aceitos por
um entregador e concluídos** em uma operação urbana do Meituan, em **8 dias de outubro de
2022** (horário local UTC+8), com duração entre 5 e 90 minutos.

Não estão representados:

- pedidos recusados ou não aceitos (`is_courier_grabbed == 0`);
- pedidos pré-agendados (`is_prebook == 1`);
- waybills com timestamps ou coordenadas sentinela;
- entregas com duração inferior a 5 min ou superior a 90 min;
- pedidos cancelados, reembolsados ou não concluídos;
- outras cidades, países, estações do ano ou regimes operacionais.

### 4.2 Contexto operacional

O dataset descreve o processo de despacho de uma plataforma de *on-demand food delivery*:
o pedido é criado, entra no sistema de despacho, é oferecido a entregadores, pode ser
recusado, e é finalmente coletado no restaurante e entregue ao cliente. O modelo é
treinado sobre o resultado observado desse processo, não sobre um experimento controlado.

O contexto de decisão é:

- **checkout:** estimativa antes de haver entregador; features `reduced`.
- **despacho/aceite:** estimativa depois que o entregador aceitou; features `complete`.

As coordenadas foram deslocadas uniformemente para preservar privacidade; as distâncias L1
são uma aproximação da estrutura relativa, não a rota real. `da_id` é uma área de negócio
que pode se sobrepor geograficamente e misturar tipos de pedido; não é um bairro
administrativo.

### 4.3 Viés de seleção

Como só entram pedidos aceitos e concluídos, a população é **condicionada ao sucesso e à
atribuição**. Isso pode enviesar o modelo para pedidos “mais fáceis” ou para regiões com
melhor oferta de entregadores. Pedidos recusados, cancelados ou muito longos — justamente
os mais críticos para a experiência — ficam de fora. O modelo não deve ser lido como
representativo de “todos os pedidos do Meituan” nem de “todos os pedidos de delivery”.

---

## 5. Métricas

### 5.1 Métricas registradas

As seis execuções de modelo registram, para treino, validação e teste:

- **MAE** (erro absoluto médio, em minutos) — métrica primária.
- **MSE** (erro quadrático médio, em min²) — base do RMSE.
- **RMSE** (raiz do erro quadrático médio, em minutos).
- **R²** (coeficiente de determinação).

**Por que MAE é a métrica primária:** ela está na mesma unidade do negócio (minutos), é
diretamente interpretável como “erro típico” e é menos sensível a outliers do que o RMSE.
Para uma promessa de prazo, o RMSE também importa porque erros grandes são
desproporcionalmente danosos; por isso ele é reportado ao lado do MAE. O R² mostra quanto
da variância do tempo de entrega o modelo explica em relação a uma baseline trivial.

**Métricas de classificação não se aplicam:** Accuracy, Precision, Recall, F1 e AUC
pressupõem classes; aqui o alvo é contínuo (minutos). O MSE é reportado como base
matemática do RMSE, não como métrica de decisão.

### 5.2 Resultados no split de teste

Valores do MLflow (`test_*`), arredondados para três casas decimais. O conjunto de teste
tem 50.000 linhas em todas as execuções. “ΔMAE” é a redução relativa do MAE em relação ao
baseline oficial no mesmo split.

| Run | MAE treino | MAE validação | MAE teste | RMSE teste | R² teste | ΔMAE vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_forest-reduced` | 5,132 | 6,084 | 6,081 | 7,824 | 0,557 | 36,0% |
| `random_forest-complete` | 3,986 | 5,609 | 5,603 | 7,206 | 0,624 | 41,1% |
| `hist_gradient_boosting-reduced` | 5,793 | 5,934 | 5,941 | 7,654 | 0,576 | 37,5% |
| **`hist_gradient_boosting-complete`** | **5,309** | **5,470** | **5,464** | **7,040** | **0,641** | **42,5%** |
| `lightgbm-reduced` | 5,900 | 5,973 | 5,976 | 7,698 | 0,571 | 37,1% |
| `lightgbm-complete` | 5,400 | 5,489 | 5,486 | 7,064 | 0,639 | 42,3% |
| `baseline-meituan` | — | — | 9,506 | 11,610 | 0,024 | referência |

O MSE também está registrado no MLflow; para o HGB `complete`, o valor de teste é
49,563 min². O baseline tem MSE de 134,784 min², inflado por erros extremos da promessa.

### 5.3 Leitura dos resultados

1. **Todos os modelos superam o baseline em MAE e R² no split de teste.** O melhor
   resultado é o HGB `complete`: MAE de 5,464 min, cerca de 18,5% do tempo médio realizado
   (29,562 min), e R² de 0,641.
2. **O ganho do estágio `complete` é consistente.** Para os três algoritmos, adicionar as
   10 features operacionais reduz o MAE em cerca de 0,5 min. Isso é esperado: o estágio tem
   informação sobre entregador, fila, carga e wave.
3. **HGB e LightGBM generalizam melhor que o Random Forest.** O gap treino→validação do
   Random Forest `complete` é de 1,62 min (3,986 → 5,609); no HGB `complete` é de 0,16 min
   (5,309 → 5,470). O Random Forest memoriza mais e tem artefato muito maior.
4. **As diferenças entre HGB e LightGBM são pequenas.** No `complete`, 0,022 min de MAE;
   no `reduced`, 0,034 min. Não há teste de significância; a escolha é pragmática.
5. **O R² de 0,641 é moderado.** O modelo explica cerca de 64% da variância no teste; ainda
   há muito erro não explicado, coerente com a ausência de clima, trânsito, oferta e
   variabilidade de preparo.

### 5.4 Por que o baseline não é um “modelo concorrente” justo

O `baseline-meituan` usa `estimate_arrived_time`, a **promessa** exibida ao consumidor. Essa
promessa inclui margem de segurança deliberada (no notebook de origem, cerca de 8–9 min de
folga). Portanto:

- O baseline é **conservador por construção**: erra mais em MAE, mas entrega menos vezes
  depois do prazo prometido.
- O modelo é uma **previsão pontual do tempo realizado**: erra menos em média, mas não tem
  margem embutida; uma previsão pontual não garante que o pedido chegue antes dela.
- Dizer que o modelo é “42,5% melhor que a Meituan” seria uma leitura indevida: o número é
  uma comparação de **MAE contra a promessa neste split**, não sobre o sistema de produção
  da Meituan nem sobre satisfação do cliente.

Essa distinção é essencial para usos de promessa ao cliente: o modelo precisa de uma etapa
de calibração/quantil antes de virar SLA.

### 5.5 Cautelas estatísticas

- Métricas de uma única split aleatória, sem intervalos de confiança, validação cruzada
  repetida ou teste pareado.
- O split não é temporal; não mede degradação ao longo do tempo.
- As métricas de teste são **otimistas** por causa do vazamento do target encoding
  (seção 6.1).
- Não há análise de subgrupos (distrito, POI, entregador, turno) nem de calibração.
- Não há teste de robustez a outliers, dados faltantes ou mudança de distribuição.

---

## 6. Limitações

### 6.1 Vazamento do target encoding (crítica)

`poi_target_enc` é calculado com `KFold(n_splits=5, shuffle=True)` sobre o **frame limpo
completo, antes do split train/test**. Para cada linha, o encoding usa a média do alvo em
outras dobras — que incluem linhas que depois cairão no teste. O teste, portanto, não é
totalmente cego ao alvo.

Consequências:

- As métricas de teste são **otimistas**; o erro real em produção tende a ser maior.
- O ranking entre modelos pode se manter, mas os valores absolutos não devem ser usados
  como estimativa de generalização.
- O comportamento é **herdado do notebook de origem** e está marcado em todas as runs com
  `known_limitation=poi_target_enc_oof_pre_split`. Não foi “corrigido” silenciosamente para
  preservar a comparabilidade com as etapas anteriores.

Para um modelo de produção: recalcular o encoding **dentro** de cada dobra de treino, depois
do split, e preferencialmente de forma temporal (sem `shuffle` entre dias). Reavaliar tudo
antes de qualquer decisão.

### 6.2 Escopo temporal e geográfico

- Apenas 8 dias de outubro de 2022, uma operação urbana anonimizada.
- Sem validação externa em outra cidade, estação, ano ou regime operacional.
- Sem análise de sazonalidade, feriados, clima ou eventos.
- Coordenadas deslocadas: não é possível geolocalizar a cidade nem validar rotas reais.

### 6.3 Viés de seleção

- Apenas pedidos imediatos, aceitos e concluídos.
- Pedidos recusados, cancelados, pré-agendados, com sentinela ou fora de 5–90 min ficam de
  fora.
- A população é condicionada ao sucesso; os casos mais difíceis podem estar
  sub-representados.

### 6.4 Covariáveis ausentes

Clima, trânsito, obras, promoções, oferta de entregadores, tempo real de preparo,
características do cliente/entregador e qualidade da rota não estão no modelo. Parte do
erro é irredutível com os dados disponíveis.

### 6.5 Previsão pontual, sem incerteza

- O modelo retorna um número, não uma distribuição.
- Não há intervalos de predição nem calibração para nível de serviço.
- Uma previsão pontual não garante que o pedido chegue antes dela; para promessa, é
  necessário um quantil superior (por exemplo, P90) e backtest de cobertura.

### 6.6 Comparação com o baseline

- O baseline é uma promessa com margem, não uma previsão pontual.
- O baseline não é truncado em 5–90 min e tem cauda longa (máx. 325 min), o que infla
  RMSE/MSE.
- A comparação é útil como referência operacional, mas não é *like-for-like*.

### 6.7 Categorias e engenharia de features

- `da_id` é usado como número; o modelo pode impor uma ordem arbitrária a categorias.
- `poi_target_enc` é a única codificação de categoria de alta cardinalidade.
- Distâncias são L1 sobre coordenadas deslocadas, não rotas reais.
- `carga_entregador`, `prior_rejections` e `pos_na_wave` são proxies operacionais, não
  medidas causais.
- `wave_start_time` tem issues conhecidas no dataset; as waves foram usadas como
  fornecidas, com 100% de cobertura.

### 6.8 Validação estatística

- Uma única split aleatória; sem validação cruzada repetida, bootstrap ou teste pareado.
- Sem intervalos de confiança para MAE/RMSE/R².
- Sem análise de poder estatístico para diferenças entre modelos.
- Sem análise de estabilidade entre sementes.

### 6.9 Explicabilidade desigual

- Random Forest e LightGBM têm `feature_importances_` logado em
  `plots/feature_importance.json`.
- HGB não expõe `feature_importances_`; o artefato registra a ausência.
- Importância nativa não é causal; não substitui SHAP/permutation importance nem análise
  de subgrupos.

### 6.10 Artefatos e serving

- Random Forest: 176–330 MB; pode ser caro em memória e latência.
- HGB: 1,8 MB; LightGBM: 0,9–1,0 MB.
- Não foi medido throughput/latência, nem testado serving, concorrência, fallback ou
  versionamento de modelo em produção.
- Pickles em `models/` usam `pickle` padrão; os modelos logados no MLflow usam
  `cloudpickle`. A compatibilidade com versões futuras de Python/sklearn/LightGBM não é
  garantida.

### 6.11 Governança ainda não implementada

- Não há monitoramento contínuo, alertas, incident response ou retreino automático.
- Não há registro formal de aprovação, auditoria ou revisão periódica.
- A seção 13 é uma **proposta** de monitoramento, não uma implementação.

---

## 7. Usos recomendados

| Uso | Por que é adequado | Condições |
| --- | --- | --- |
| Benchmark e seleção de modelo | Compara três algoritmos e dois estágios no mesmo split, com rastreabilidade. | Usar apenas como referência interna; não generalizar para produção sem validação externa. |
| Apoio à decisão humana | Fornece uma estimativa de duração com erro médio de ~5–6 min em pedidos comparáveis. | Analista revisa; exibir incerteza/faixa; não automatizar consequências. |
| Priorização operacional | Ajuda a identificar pedidos/regiões com maior tempo esperado. | Usar em nível agregado; não penalizar indivíduos; validar impacto. |
| Simulação e planejamento | Permite cenários “e se” sobre features operacionais. | Cenários dentro da distribuição observada; não tratar como causal. |
| Base para um futuro modelo de produção | Pipeline, features e métricas já são reproduzíveis. | Corrigir leakage, calibrar, validar temporalmente, medir latência, monitorar. |
| Ensino e reprodutibilidade | Demonstra DVC + MLflow + model card de ponta a ponta. | Deixar claro que é artefato acadêmico/experimental. |
| Referência para monitoramento | Métricas e run IDs servem de baseline para drift. | Atualizar a referência após retreino; não usar valores otimistas como SLA. |

**Usuários pretendidos:** cientistas de dados, analistas de operações e revisores de
governança, com supervisão. **Não** é destinado a consumidores finais nem a decisões
automatizadas sem revisão.

---

## 8. Usos não recomendados

| Uso | Por que não |
| --- | --- |
| Definir a promessa ao cliente diretamente | É uma previsão pontual, sem margem nem calibração; uma previsão pontual não garante que o pedido chegue antes dela. Use um quantil superior e backtest. |
| Avaliar, punir ou premiar entregadores | O alvo é o tempo do sistema, não o desempenho individual; há confundimento por distrito, loja, clima e oferta. Features como recusas e carga são operacionais, não métricas de mérito. |
| Despacho automático | O modelo prevê duração, não o valor causal de uma atribuição; não foi validado como política de decisão. |
| Generalizar para outra cidade, país, estação ou ano | Sem validação externa; a distribuição e a operação podem mudar radicalmente. |
| Decisões de crédito, preço, seguro, RH ou elegibilidade | Fora do escopo; risco de discriminação e de uso indevido de dados de localização. |
| Substituir o baseline oficial como “verdade” | O baseline é uma promessa; o modelo é uma estimativa. Nenhum dos dois é ground truth. |
| Uso comercial dos dados ou derivados | A licença CC BY-NC 4.0 proíbe uso comercial e redistribuição dos dados brutos. |
| Tempo real sem teste de latência/fallback | Não há serving, medição de p95, concorrência, cache ou plano de falha. |
| Decisões com efeito legal ou sobre direitos | Não há base legal, auditoria de fairness nem mecanismo de contestação. |
| Reidentificação de pessoas ou estabelecimentos | Os IDs são pseudonimizados e as coordenadas deslocadas; tentar reidentificar viola privacidade e a licença. |

---

## 9. Riscos

| Risco | Causa | Impacto | Mitigação |
| --- | --- | --- | --- |
| Subestimar o tempo e prometer cedo demais | Previsão pontual sem margem; erro de ~5–6 min e cauda longa. | Atraso percebido, reclamações, perda de confiança. | Usar quantil superior/P90, margem de segurança, backtest de cobertura e monitoramento de atrasos. |
| Superestimar o tempo e criar folga excessiva | Modelo conservador ou distribuição deslocada. | Entregadores ociosos, custo maior, promessa pouco competitiva. | Calibrar; monitorar viés médio e utilização; revisar por região. |
| Selecionar modelo com base em métricas infladas | Vazamento do target encoding antes do split. | Falsa confiança, escolha errada, degradação em produção. | Corrigir encoding dentro das dobras; reavaliar com split temporal; não usar os números atuais como SLA. |
| Degradação por data drift | Mudança de demanda, clima, oferta, preços, comportamento. | MAE/RMSE sobem; promessa piora. | Monitorar PSI/KS, MAE por janela, retreinar com gatilhos. |
| Viés geográfico ou por estabelecimento | Distritos, POIs e entregadores têm distribuições diferentes. | Serviço pior em áreas vulneráveis; inequidade. | Métricas por subgrupo, auditoria de fairness, revisão de features e políticas. |
| Feedback loop | Decisões baseadas no modelo alteram os dados futuros. | O modelo pode se autoconfirmar e perpetuar vieses. | Registrar decisões, auditorias aleatórias, avaliação causal, revisão humana. |
| Vigilância de entregadores | Features de recusa, carga e wave podem ser lidas como desempenho. | Pressão indevida, precarização, violação de dignidade. | Política explícita de não uso individual; agregação; revisão ética e sindical. |
| Privacidade e reidentificação | Localização, POI, distrito e horários são sensíveis. | Exposição de pessoas ou estabelecimentos. | Minimização, acesso restrito, agregação, não tentar reidentificar, cumprir a licença. |
| Viés de automação | Usuários confiam demais no número. | Decisões ruins sem contestação. | Exibir incerteza, treinar usuários, manter revisão humana e canal de contestação. |
| Modelo obsoleto ou não versionado | Retreino sem registro, mudança de features, pickle antigo. | Resultados irreprodutíveis; incidentes. | DVC + MLflow + model card versionado; revisão a cada retreino. |
| Violação de licença | Redistribuir dados ou usar comercialmente. | Problema legal e ético. | Manter dados fora do Git; uso não comercial; citar Meituan. |

---

## 10. Aspectos éticos

### 10.1 Privacidade e proteção de dados

- Os dados já vêm pseudonimizados e com coordenadas deslocadas; ainda assim, localização,
  horário, POI e distrito são dados sensíveis.
- O repositório não redistribui os dados brutos; o DVC remote é local e o MLflow registra
  hashes, não o conteúdo das linhas.
- Não se deve tentar reidentificar pessoas, entregadores ou estabelecimentos.
- Acesso aos dados e artefatos deve ser restrito a quem precisa; o model card não deve ser
  usado para burlar essa restrição.

### 10.2 Justiça e equidade

- O modelo não usa atributos protegidos, mas geografia, POI e distrito podem funcionar como
  proxies de renda, raça ou vulnerabilidade.
- Não há análise de fairness por subgrupo. Antes de qualquer uso operacional, medir MAE,
  viés e taxa de atraso por distrito, POI e perfil de entregador.
- Diferenças de erro entre regiões podem amplificar inequidades se o modelo orientar
  priorização ou promessa.

### 10.3 Trabalho e dignidade

- Features como `prior_rejections`, `carga_entregador` e `pos_na_wave` descrevem o
  processo, não o mérito do entregador.
- O modelo **não** deve ser usado para avaliar, punir, premiar ou ranquear entregadores.
- Qualquer uso que afete renda, escala ou condições de trabalho exige negociação,
  transparência e revisão humana.

### 10.4 Transparência e accountability

- O pipeline é reproduzível por DVC; as execuções são rastreáveis por MLflow; as limitações
  estão documentadas neste card.
- O responsável pelo modelo está nomeado na seção 11.
- Ainda não há processo formal de aprovação, auditoria independente ou contestação; isso
  deve preceder qualquer uso externo.

### 10.5 Licença e uso não comercial

- O uso dos dados e de artefatos derivados deve observar a CC BY-NC 4.0; uso comercial não
  é permitido.
- Publicações devem incluir: “This research was supported by data provided by Meituan.”
- O model card não concede licença adicional sobre os dados.

### 10.6 Supervisão humana

- O modelo é de apoio, não de decisão autônoma.
- Usuários devem ver a estimativa, a incerteza e o contexto; devem poder discordar e
  escalar.
- Decisões com impacto sobre clientes ou entregadores exigem revisão humana e registro.

---

## 11. Responsável pelo modelo

| Papel | Nome / contato | Responsabilidade |
| --- | --- | --- |
| Responsável técnico | Guilherme (`guinbraga`), autor do repositório e executor das runs (`mlflow.user=guilherme`) | Escopo, treino, avaliação, documentação, versionamento e revisão deste card. |
| Contato | Repositório Git do projeto (issues/commits) | Receber dúvidas, correções e solicitações de revisão. |
| Aprovação para produção | **Não concedida** | Este card documenta um artefato experimental; qualquer uso externo exige nova aprovação. |
| Revisão | Antes de qualquer uso externo; após retreino, mudança de dados/features ou incidente; no mínimo semestral | Atualizar métricas, limitações, riscos e status. |

Este card não substitui aprovação formal. No contexto acadêmico, a validação final cabe ao
orientador/avaliador da disciplina; em um contexto organizacional, cabe ao comitê de
governança de IA e ao owner do produto.

---

## 12. Rastreabilidade e reprodução

### 12.1 Onde consultar no MLflow

O servidor em execução responde em **`http://localhost:5001`**. O experimento é
`meituan-eta` (`experiment_id=1`), com **7 execuções ativas**: seis modelos e o baseline.

- Visão do experimento: `http://localhost:5001/#/experiments/1`
- Uma execução: `http://localhost:5001/#/experiments/1/runs/<run_id>`
- Runs e IDs: seção 2.2 deste card.

Para consultar por código:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
runs = mlflow.search_runs(experiment_names=["meituan-eta"])
print(
    runs[
        [
            "tags.mlflow.runName",
            "run_id",
            "tags.data_version",
            "metrics.test_mae",
            "metrics.test_rmse",
            "metrics.test_r2",
        ]
    ].sort_values("metrics.test_mae")
)
```

Se o servidor não estiver no ar, a partir da raiz do repositório:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

### 12.2 O que cada run registra

- **Parâmetros:** algoritmo, estágio, seed, tamanho de treino/validação, número e lista de
  features, hiperparâmetros e `data_sha256`.
- **Métricas:** `{train,val,test}_{mae,mse,rmse,r2}` para os seis modelos;
  `test_{mae,mse,rmse,r2}` para o baseline.
- **Tags:** `mlflow.runName`, `data_version` (12 hex), `known_limitation` e metadados de
  origem Git.
- **Dataset input:** `mlflow.data.from_pandas` com source, nome, target e contexto
  (`training`/`testing`), além do digest do MLflow.
- **Artefatos:** `plots/residuals.png`, `plots/feature_importance.png`,
  `plots/feature_importance.json`.
- **Modelo:** cada run de modelo tem pelo menos um `LoggedModel` do MLflow 3 (o baseline
  não tem). Os pickles também estão em `models/`, versionados pelo DVC.

### 12.3 Arquivos de governança

| Arquivo | Papel |
| --- | --- |
| [`dvc.yaml`](../../dvc.yaml) | Define os sete estágios e seus deps/outs/params. |
| [`dvc.lock`](../../dvc.lock) | Congela hashes md5 de dados, código e artefatos. |
| [`params.yaml`](../../params.yaml) | Hiperparâmetros, seed, split e configuração do MLflow. |
| [`reports/experiment_registry.md`](../../reports/experiment_registry.md) | Tabela legível das 7 execuções. |
| [`reports/experiment_registry.csv`](../../reports/experiment_registry.csv) | Versão machine-readable do registry. |
| [`reports/metrics.json`](../../reports/metrics.json) | Métricas de teste por modelo e baseline. |
| [`README.md`](../../README.md) | Instruções de reprodução, estágios e limitações. |
| `models/*.pkl` | Pickles dos seis modelos (DVC out; não versionados no Git). |
| `mlflow.db` + `mlruns/` | Store e artefatos do MLflow (gitignored; não redistribuídos). |

### 12.4 Como reproduzir

```bash
# Pipeline completo (bootstrap → validate → prepare → features → split → train → evaluate)
dvc repro

# Métricas e estado
dvc metrics show
dvc status
dvc dag
```

O `dvc repro` exige os quatro arquivos brutos em `data/` e pode levar de 10 a 25 minutos.
Para consultar este card, **não é necessário retreinar**: use o MLflow e os relatórios.

### 12.5 Nota de estado

Este card foi escrito a partir das execuções já registradas; não foi gerado por um novo
treino. Se o workspace tiver sido alterado depois, verifique `dvc status` e compare
`run_id`, `data_version` e `start_time` antes de reutilizar os números.

---

## 13. Monitoramento (proposta para a Etapa 4)

> **Não implementado.** Esta seção responde à pergunta “como saberíamos que o modelo
> continua funcionando?” com indicadores e gatilhos que podem ser implementados em uma
> próxima etapa.

### 13.1 Dados

| Indicador | O que medir | Referência / alerta inicial |
| --- | --- | --- |
| Valores faltantes | % de nulos/sentinelas por feature | > 1% ou 2× a taxa de referência |
| Distribuição | PSI por feature; KS para contínuas | PSI > 0,2; KS com p < 0,01 |
| Qualidade | distâncias negativas, hora fora de 0–23, `da_id`/`poi_id` novos, duplicatas | Qualquer violação de schema/range |
| Cobertura | % de waves casadas; sentinelas 0 | Queda > 2 p.p. vs referência |
| Target | distribuição de duração por janela | Mudança > 10% na média/mediana |

### 13.2 Modelo

| Indicador | O que medir | Referência / alerta inicial |
| --- | --- | --- |
| MAE | MAE por janela (dia/semana) com rótulos disponíveis | > 6,5 min por 2 janelas consecutivas (HGB `complete`: 5,46) |
| RMSE | RMSE por janela | > 8,5 min (referência: 7,04) |
| R² | R² por janela | < 0,55 (referência: 0,64) |
| Viés | média (real − previsto) | valor absoluto > 0,5 min |
| Cauda | P90/P95 do erro absoluto | P90 > 14 min (a calibrar a partir do teste) |
| Subgrupos | MAE por distrito, POI, turno, carga | Gap > 2 min entre subgrupos |
| Calibração | cobertura de intervalos, se implementados | Cobertura fora de ±5 p.p. do nominal |

### 13.3 Negócio

| Indicador | O que medir | Por que importa |
| --- | --- | --- |
| Acurácia da promessa | % de pedidos com `arrive_time <= estimate_arrived_time` | Mede o efeito real na experiência. |
| Atraso | minutos de atraso vs promessa; % de atrasos > 10 min | Captura a cauda, não só a média. |
| Reclamações | reclamações por 1.000 pedidos; motivos | Sinal direto de dano ao cliente. |
| Cancelamentos | taxa após a promessa | Efeito de promessas ruins. |
| Utilização | pedidos por entregador/hora; ociosidade | Custo operacional. |
| Custo | custo por entrega; compensações | Impacto financeiro. |
| Utilização do modelo | % de pedidos pontuados; fallback; latência p95 | Saúde do serving. |

### 13.4 Governança

| Indicador | O que medir | Ação |
| --- | --- | --- |
| Incidentes | número, severidade, tempo de resposta | Registrar e revisar em 5 dias úteis. |
| Reclamações | volume e temas | Triagem e análise de causa raiz. |
| Vieses | auditoria por subgrupo | Revisão de features/política se gap relevante. |
| Revisão | status do card, aprovação, última revisão | Bloquear uso se o card estiver vencido. |
| Licença | uso não comercial; dados fora do Git | Auditoria periódica. |
| Modelo em uso | versão, run_id, data_version | Inventário e rollback. |

### 13.5 Gatilhos de retreino/revisão

- MAE > 6,5 min por duas janelas consecutivas, ou aumento > 20% vs referência.
- PSI > 0,2 em features-chave (distância, hora, preparo, demanda, carga).
- Mudança de cidade, estação, política de despacho ou mix de pedidos.
- Incidente com dano a cliente/entregador ou reclamação recorrente.
- Mudança de schema, fonte de dados ou pipeline de features.
- Revisão periódica semestral, mesmo sem alerta.

### 13.6 Resposta a incidentes (proposta)

1. Detectar e registrar o incidente com `run_id`, `data_version` e janela afetada.
2. Suspender uso automatizado; manter o modo assistido ou fallback.
3. Comparar métricas com a referência e identificar drift/dados/feature.
4. Corrigir, reavaliar e, se necessário, retreinar com nova versão.
5. Aprovar novamente antes de voltar ao uso; atualizar este card.

---

## 14. Mapa de conformidade com a Etapa 4

| Requisito do PDF | Onde está neste card |
| --- | --- |
| Objetivo | Seção 1 |
| Dados utilizados | Seção 3 |
| População/contexto | Seção 4 |
| Métricas | Seção 5 |
| Limitações | Seção 6 |
| Usos recomendados | Seção 7 |
| Usos não recomendados | Seção 8 |
| Riscos | Seção 9 |
| Aspectos éticos | Seção 10 |
| Responsável pelo modelo | Seção 11 |
| Versionamento/reprodutibilidade (contexto) | Seções 2, 3.2 e 12 |
| Monitoramento (contexto) | Seção 13 |

---

## 15. Versões e manutenção

| Versão | Data | Mudança | Responsável |
| --- | --- | --- | --- |
| 1.0 | 2026-11-21 | Card inicial cobrindo os seis modelos da Etapa 3 e o baseline. | Guilherme (`guinbraga`) |

**Quando revisar:**

- após qualquer retreino ou mudança de hiperparâmetro/feature;
- após mudança nos dados brutos ou no pipeline de preparação;
- após incidente, reclamação relevante ou alerta de drift;
- antes de qualquer uso externo, publicação ou apresentação;
- no mínimo a cada semestre.

Ao revisar, comparar `run_id`, `data_version`, `start_time` e métricas com o MLflow; se os
números mudarem, atualizar as tabelas e a versão deste card.

---

## 16. Referências

- [Enunciado do Projeto Final — Etapas 3 e 4](<../ProjetoFinal_Etapas3_4.pdf>)
- [README do repositório](../../README.md)
- [Registro dos experimentos (Markdown)](../../reports/experiment_registry.md)
- [Registro dos experimentos (CSV)](../../reports/experiment_registry.csv)
- [Métricas de teste](../../reports/metrics.json)
- [Pipeline DVC](../../dvc.yaml) e [lock](../../dvc.lock)
- [Parâmetros](../../params.yaml)
- [Notebook de origem](../../notebooks/meituan_delivery_prediction_v3.md)
- [Relatório de dados do desafio (15/10/2024)](<../tsl_meituan_2024_data_report_20241015.pdf>)
- [Background e descrição dos dados (21/03/2024)](<../TSL-Meituan challenge_background and data_20240321.pdf>)
- [Licença dos dados](../../License.txt)
- Mitchell, M. et al. *Model Cards for Model Reporting*. FAT* 2019.
  <https://doi.org/10.1145/3287560.3287596>
- MLflow — Tracking e Model Registry: <https://mlflow.org/docs/latest/ml/tracking/>

---

## Apêndice A — Ficha rápida por execução

| Run | Momento | n features | MAE teste | RMSE teste | R² teste | Tamanho | Uso indicado |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `random_forest-reduced` | Checkout | 9 | 6,081 | 7,824 | 0,557 | 176,2 MB | Benchmark; não é a escolha primária. |
| `random_forest-complete` | Despacho | 19 | 5,603 | 7,206 | 0,624 | 329,6 MB | Benchmark/interpretabilidade; artefato grande. |
| `hist_gradient_boosting-reduced` | Checkout | 9 | 5,941 | 7,654 | 0,576 | 1,8 MB | **Escolha primária no checkout.** |
| `hist_gradient_boosting-complete` | Despacho | 19 | 5,464 | 7,040 | 0,641 | 1,8 MB | **Escolha primária no despacho.** |
| `lightgbm-reduced` | Checkout | 9 | 5,976 | 7,698 | 0,571 | 0,9 MB | Alternativa compacta com importância nativa. |
| `lightgbm-complete` | Despacho | 19 | 5,486 | 7,064 | 0,639 | 1,0 MB | Alternativa quase equivalente; importância nativa. |
| `baseline-meituan` | Referência | — | 9,506 | 11,610 | 0,024 | — | Referência oficial; não é modelo treinado. |

## Apêndice B — Glossário rápido

| Termo | Definição |
| --- | --- |
| Waybill | Uma tentativa de entrega de um pedido; um pedido pode gerar vários waybills. |
| Checkout | Momento da compra, antes de existir entregador alocado. |
| Despacho/aceite | Momento em que o entregador aceita a atribuição. |
| Target encoding | Substituição de uma categoria (aqui, `poi_id`) pela média do alvo, com suavização. |
| OOF | *Out-of-fold*: encoding calculado em dobras, usando o alvo de outras dobras. |
| MAE | Erro absoluto médio, em minutos. |
| RMSE | Raiz do erro quadrático médio, em minutos; penaliza erros grandes. |
| R² | Proporção da variância do alvo explicada pelo modelo. |
| Baseline | Referência oficial `estimate_arrived_time`, uma promessa com margem. |
| Data drift | Mudança na distribuição dos dados de entrada ao longo do tempo. |
| Model drift | Degradação do desempenho do modelo ao longo do tempo. |
| LoggedModel | Entidade de modelo do MLflow 3, separada dos artefatos clássicos da run. |
