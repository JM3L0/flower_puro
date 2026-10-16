# 🌸 Tutorial Didático de Aprendizado Federado (Flower + PyTorch)

Este repositório é um guia prático, canônico e autocontido para estudar e executar **Aprendizado Federado (*Federated Learning*)** utilizando o framework oficial [Flower](https://flower.ai/) e o motor de Deep Learning [PyTorch](https://pytorch.org/) sobre o dataset CIFAR-10.

---

## 🛠️ Especificações Técnicas do Motor de Execução

O ambiente foi projetado para seguir rigorosamente as convenções da literatura acadêmica e a arquitetura moderna do Flower 1.x:

### 1. Motor de Deep Learning (PyTorch)
* **Framework**: PyTorch (`torch >= 2.2.0`, `torchvision >= 0.17.0`).
* **Dispositivo de Cálculo**: Seleção automática de GPU NVIDIA via CUDA (`cuda:0` / cuDNN) quando disponível, com *fallback* automático para CPU.
* **Dataset**: **CIFAR-10** (60.000 imagens coloridas $32 \times 32$, 10 classes: *Avião, Automóvel, Pássaro, Gato, Cervo, Cachorro, Sapo, Cavalo, Navio, Caminhão*).
* **Pré-processamento**: Conversão para tensor e normalização de canal com média $(0.5, 0.5, 0.5)$ e desvio padrão $(0.5, 0.5, 0.5)$, mapeando pixels para o intervalo $[-1.0, 1.0]$.
* **Arquitetura da Rede Neural (CNN)**:
  * `Conv2d(3, 6, kernel_size=5)` + `ReLU` + `MaxPool2d(2, 2)` $\to$ saída $14 \times 14 \times 6$
  * `Conv2d(6, 16, kernel_size=5)` + `ReLU` + `MaxPool2d(2, 2)` $\to$ saída $5 \times 5 \times 16 = 400$ ativações
  * `Linear(400, 120)` + `ReLU`
  * `Linear(120, 84)` + `ReLU`
  * `Linear(84, 10)` $\to$ saída de 10 *logits* não-normalizados.
* **Função de Custo**: *Cross-Entropy Loss* (`nn.CrossEntropyLoss`).
* **Otimizador Local**: Estocástico com Momentum (`torch.optim.SGD`, $\text{momentum}=0.9$, $\text{lr}=0.01$).

---

### 2. Orquestração e Protocolo do Flower (Flower 1.x)
* **Framework**: Flower (`flwr >= 1.15.0`).
* **Paradigma de Componentes**:
  * **`ClientApp`**: Instanciado via fábrica `client_fn(context: Context)`, criando objetos `NumPyClient`.
    * `get_parameters()`: Serializa tensores do `state_dict()` do PyTorch em listas de matrizes NumPy.
    * `fit(parameters, config)`: Desserializa pesos globais, treina localmente por $E$ épocas e devolve os parâmetros atualizados com o número de amostras locais.
    * `evaluate(parameters, config)`: Mede perda e acurácia sobre os dados locais de validação.
  * **`ServerApp`**: Orquestrado via `server_fn(context: Context)` retornando `ServerAppComponents`.
    * Mantém o modelo global e gerencia as rodadas federadas.
    * `evaluate_fn`: Avalia o modelo global a cada rodada diretamente contra o conjunto de teste centralizado (10.000 imagens).
    * `fit_metrics_aggregation_fn`: Consolida perdas de treino ponderadas pelo volume de dados de cada nó.
* **Estratégias de Agregação Implementadas**:
  * **`FedAvg` (Federated Averaging - McMahan et al., 2017)**: Média ponderada clássica:
    $$\mathbf{W}_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{W}_{t+1}^k$$
  * **`FedProx` (Li et al., 2020)**: Adiciona termo proximal $\frac{\mu}{2} \|\mathbf{w} - \mathbf{w}^t\|^2$ para estabilizar clientes sob alta heterogeneidade.
  * **`FedMedian` (Yin et al., 2018)**: Mediana coordenada por coordenada, robusta contra clientes discrepantes.

---

### 3. Mecanismo de Particionamento de Dados (`flwr-datasets`)
* **Biblioteca**: `flwr-datasets[vision] >= 0.5.0` integrada com Hugging Face Datasets.
* **Heterogeneidade Configurável (IID vs. Não-IID)**:
  * **IID (`IidPartitioner`)**: Divisão uniforme e balanceada das classes entre todos os clientes ($\approx 500$ imagens/classe por nó).
  * **Não-IID Dirichlet (`DirichletPartitioner`)**: Simula distribuições desbalanceadas do mundo real através da distribuição $\text{Dir}(\alpha)$:
    * $\alpha = 100.0$: Distribuição homogênea equivalente a IID.
    * $\alpha = 1.0$: Assimetria suave entre clientes.
    * $\alpha = 0.5$: Assimetria moderada (clientes concentram dados em 2 a 3 classes).
    * $\alpha = 0.1$: Assimetria severa (*Névoa de Guerra* federada, cada cliente possui quase que exclusivamente 1 classe).

---

## 📊 Comparativo dos Modos de Execução

| Modo | Arquivo / Comando | Onde Executa | Aceleração | Indicado Para |
|---|---|---|:---:|---|
| **Google Colab** | `executar_no_colab.ipynb` | Nuvem (Google Colab) | GPU T4 Gratuita | Experimentos imediatos, sem instalar nada no PC |
| **Simulação Direta** | `simulacao_direta.py` | Máquina Local / Terminal | GPU Local ou CPU | Prototipagem rápida, scripts bash, gráficos automáticos |
| **Flower CLI 1.x** | `flwr run .` | Máquina Local / SuperLink | Multiprocesso Ray | Padrão oficial de produção do ecossistema Flower |

---

## 🚀 Como Executar

### Opção 1: No Google Colab (Recomendado - GPU T4 Gratuita)

Abra o notebook autocontido diretamente no Google Colab:
1. Abra [colab.research.google.com](https://colab.research.google.com/).
2. Faça o upload do arquivo [`executar_no_colab.ipynb`](executar_no_colab.ipynb) (ou abra via GitHub).
3. Ative a GPU: **Ambiente de execução -> Alterar tipo de ambiente de execução -> Acelerador de hardware: T4 GPU -> Salvar**.
4. Ajuste os parâmetros no formulário interativo e clique em **Executar tudo** (`Ctrl + F9`).
5. Ao término, os gráficos de convergência de Acurácia e Perda (*Loss*) são exibidos e o arquivo `resultados_flower_puro.zip` (com os pesos do modelo `.pt`) é baixado no seu navegador.

---

### Opção 2: Localmente via Script Direto (`simulacao_direta.py`)

Esta opção dispensa daemons e executa o ciclo federado completo diretamente:

```bash
# 1. Instale as dependências
pip install -r requirements.txt

# 2. Execute com FedAvg padrão (10 clientes, 10 rodadas, dados homogêneos IID)
python simulacao_direta.py --strategy FedAvg --rounds 10 --clients 10 --alpha 100.0

# 3. Ou execute sob assimetria Não-IID (Dirichlet α=0.5)
python simulacao_direta.py --strategy FedAvg --rounds 10 --clients 10 --alpha 0.5
```

Parâmetros disponíveis no terminal:
* `--strategy`: Estratégia de agregação (`FedAvg`, `FedMedian`, `FedProx`).
* `--rounds`: Quantidade de rodadas federadas (padrão: 10).
* `--clients`: Quantidade de clientes locais (padrão: 10).
* `--epochs`: Épocas de treino local por rodada (padrão: 2).
* `--lr`: Taxa de aprendizado local SGD (padrão: 0.01).
* `--batch_size`: Tamanho do lote local (padrão: 64).
* `--alpha`: Grau de heterogeneidade Dirichlet (`100.0` para IID, `0.5` para Não-IID).
* `--output_dir`: Pasta onde gráficos PNG, tabelas Markdown e checkpoints `.pt` são salvos.

---

### Opção 3: Localmente via CLI Oficial do Flower 1.x (`flwr run`)

Se você utiliza o ecossistema Flower padrão com SuperNodes:

```bash
# Instala o pacote em modo editável
pip install -e .

# Executa a simulação configurada no pyproject.toml
flwr run .
```

---

## 📁 Estrutura de Arquivos

```text
flower_puro/
│
├── README.md                      # Este manual técnico completo
├── requirements.txt                # Dependências pip necessárias
├── pyproject.toml                  # Configuração padrão do Flower 1.x
├── .gitignore                      # Arquivos e pastas ignorados pelo Git
├── simulacao_direta.py             # Executor direto com FedAvg, gráficos e tabelas
├── executar_no_colab.ipynb         # Notebook 100% autocontido para Google Colab com GPU T4
│
└── flower_puro/                    # Pacote Python canônico do Flower
    ├── __init__.py                 # Inicializador do pacote
    ├── task.py                     # Rede CNN, carregamento CIFAR-10 (IID e Dirichlet), train e test
    ├── client_app.py               # ClientApp oficial do Flower (NumPyClient)
    └── server_app.py               # ServerApp oficial (FedAvg, FedProx, FedMedian)
```

---

## 🧠 Fluxo de Comunicação e Ciclo de Vida do Treinamento

```text
       ┌────────────────────────────────────────────────────────┐
       │                Servidor Central (ServerApp)            │
       │  1. Mantém os pesos globais do modelo W_global         │
       │  2. Avalia acurácia no conjunto de teste completo      │
       └──────────────────────────┬─────────────────────────────┘
                                  │
          Envia W_global          │          Recebe W_local
         para cada cliente        ▼          e número de amostras n_k
   ┌──────────────────────────────┴──────────────────────────────┐
   ▼                                                             ▼
┌───────────────────────────┐                 ┌───────────────────────────┐
│     Cliente 0 (ClientApp) │                 │     Cliente N (ClientApp) │
│ - Dados locais privados   │       ...       │ - Dados locais privados   │
│ - Treina modelo por E     │                 │ - Treina modelo por E     │
│   épocas locais via SGD   │                 │   épocas locais via SGD   │
│ - Retorna W_local_0       │                 │ - Retorna W_local_N       │
└───────────────────────────┘                 └───────────────────────────┘
```

1. **Broadcast**: O servidor envia os parâmetros atuais $\mathbf{W}_t$ aos clientes.
2. **Treinamento Local**: Cada cliente inicializa seu modelo PyTorch com $\mathbf{W}_t$, calcula a perda $\mathcal{L}(\mathbf{W})$ e otimiza os pesos localmente por $E$ épocas via SGD.
3. **Comunicação Segura**: Apenas matrizes de pesos atualizadas $\mathbf{W}_{t+1}^k$ e metadados de contagem $n_k$ são transmitidos. Nenhum dado bruto de imagem sai do cliente.
4. **Agregação Federada**: O servidor combina as atualizações e produz o novo modelo $\mathbf{W}_{t+1}$.
5. **Avaliação Global**: O modelo atualizado é testado contra o dataset de teste para mensurar convergência.

---

## 📤 Como Transformar em Repositório GitHub Próprio

Para publicar esta pasta como um repositório GitHub independente para o seu projeto:

```bash
cd flower_puro
git init
git add .
git commit -m "feat: initial commit - canonical Flower tutorial"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/flower-tutorial.git
git push -u origin main
```
