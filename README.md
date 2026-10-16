# 🌸 Tutorial Didático de Aprendizado Federado (Flower + PyTorch)

Este repositório é um guia prático, canônico e autocontido para estudar, executar e estender **Aprendizado Federado (*Federated Learning - FL*)** utilizando a arquitetura moderna do framework oficial [Flower 1.x](https://flower.ai/) em conjunto com o [PyTorch](https://pytorch.org/) sobre o dataset **CIFAR-10**.

---

## 📑 Sumário

1. [Visão Geral e Lógica do Aprendizado Federado](#-visão-geral-e-lógica-do-aprendizado-federado)
2. [Estrutura do Projeto e Módulos](#-estrutura-do-projeto-e-módulos)
3. [Fluxo de Dados e Conversão de Tensores](#-fluxo-de-dados-e-conversão-de-tensores)
4. [Estratégias de Agregação e Particionamento](#-estratégias-de-agregação-e-particionamento)
5. [Como o Projeto Roda no Google Colab](#-como-o-projeto-roda-no-google-colab)
6. [Guia de Alterações no Código (Repositório Local e Colab)](#-guia-de-alterações-no-código)
   - [Como alterar a Rede Neural (CNN / ResNet)](#1-alterando-a-arquitetura-da-rede-neural)
   - [Como alterar o Otimizador e Hiperparâmetros de Treino](#2-alterando-otimizador-e-hiperparâmetros)
   - [Como trocar ou adicionar Estratégias de Agregação](#3-alterando-a-estratégia-de-agregação)
   - [Como trocar o Dataset](#4-trocando-o-dataset)
   - [Como sincronizar alterações feitas no Colab de volta ao Git](#5-sincronizando-alterações-colab--git)
7. [Modos de Execução Disponíveis](#-modos-de-execução-disponíveis)

---

## 🧠 Visão Geral e Lógica do Aprendizado Federado

O **Aprendizado Federado** permite que múltiplos nós (clientes, celulares, hospitais ou servidores) treinem colaborativamente um modelo global compartilhado de Inteligência Artificial **sem jamais centralizar seus dados privados**.

### Ciclo de Treinamento Federado (Passo a Passo)

```text
       ┌────────────────────────────────────────────────────────┐
       │                Servidor Central (ServerApp)            │
       │  1. Mantém os pesos globais do modelo W_global         │
       │  2. Avalia acurácia no conjunto de teste completo      │
       └──────────────────────────┬─────────────────────────────┘
                                  │
          1. Broadcast de W_t     │          3. Envio de W_local
         para os nós clientes     ▼          e quantidade de amostras n_k
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

1. **Broadcast (Servidor $\to$ Clientes)**: O servidor central envia os parâmetros atuais do modelo global ($\mathbf{W}_t$) aos clientes participantes da rodada.
2. **Treinamento Local (Clientes)**: Cada cliente inicializa sua rede neural com os pesos recebidos e realiza $E$ épocas de treinamento estocástico (SGD) utilizando **apenas seus dados privados**.
3. **Reporte Seguro (Clientes $\to$ Servidor)**: Cada cliente envia de volta ao servidor **somente os novos pesos** ($\mathbf{W}_{t+1}^k$) e a quantidade de amostras locais processadas ($n_k$). Imagens ou rótulos privados **nunca saem do dispositivo**.
4. **Agregação Federada (Servidor)**: O servidor combina as atualizações locais por meio de uma estratégia matemática (como o `FedAvg`, calculando a média ponderada pelo volume de dados de cada nó) para gerar o novo modelo global $\mathbf{W}_{t+1}$.
5. **Avaliação Global (Servidor)**: O servidor avalia a acurácia e a perda do modelo recém-agregado em um conjunto de teste de validação global para mensurar o aprendizado.

---

## 📁 Estrutura do Projeto e Módulos

```text
flower_puro/
│
├── .gitignore                         # Arquivos ignorados (pesos .pt, dados locais, checkpoints, zip)
├── requirements.txt                   # Dependências pip necessárias
├── pyproject.toml                     # Configuração oficial do pacote e Flower 1.x CLI
├── README.md                          # Manual técnico e didático completo
│
├── executar_flower_puro.ipynb         # Notebook 100% autocontido para Google Colab com GPU T4
├── simulacao_direta.py                # Script autônomo local (CLI) com geração de gráficos e relatórios
│
└── flower_puro/                       # Pacote Python canônico do Flower
    ├── __init__.py                    # Identificador do pacote e versão (1.0.0)
    ├── task.py                        # Rede CNN, loaders de dados CIFAR-10, conversão de pesos e train/test
    ├── client_app.py                  # Componente do Cliente (ClientApp - NumPyClient)
    └── server_app.py                  # Componente do Servidor (ServerApp - FedAvg/FedProx/FedMedian)
```

### Papel de Cada Arquivo

* **[`flower_puro/task.py`](file:///c:/Users/jsous/Desktop/flower_puro/flower_puro/task.py)**:
  * Define a classe `Net` (arquitetura CNN de 2 camadas convolucionais e 3 densas).
  * Contém as funções utilitárias `get_weights(net)` e `set_weights(net, parameters)` que traduzem parâmetros entre tensores PyTorch e listas de arrays NumPy do Flower.
  * Realiza o particionamento do dataset CIFAR-10 usando `flwr-datasets` (distribuição IID ou Dirichlet Não-IID) com cache em memória para ganho de performance.
  * Fornece os loops canônicos de treino local (`train`) e avaliação de acurácia/perda (`test`).
* **[`flower_puro/client_app.py`](file:///c:/Users/jsous/Desktop/flower_puro/flower_puro/client_app.py)**:
  * Implementa a classe `FlowerClient(NumPyClient)` com os métodos padrão:
    * `get_parameters()`: Devolve os pesos locais do modelo.
    * `fit(parameters, config)`: Carrega os pesos globais, executa o treino local e devolve os pesos atualizados, número de amostras e métricas.
    * `evaluate(parameters, config)`: Avalia o modelo nos dados locais de validação do cliente.
  * Expõe a fábrica `client_fn(context)` e a instância oficial `app = ClientApp(client_fn=client_fn)`.
* **[`flower_puro/server_app.py`](file:///c:/Users/jsous/Desktop/flower_puro/flower_puro/server_app.py)**:
  * Orquestra as rodadas federadas através da fábrica `server_fn(context)`.
  * Cria o callback `evaluate_fn` para mensurar a acurácia global do modelo diretamente no conjunto de teste completo (10.000 imagens).
  * Agrega as métricas locais via `weighted_average`.
  * Instancia a estratégia selecionada (`FedAvg`, `FedProx` ou `FedMedian`).
  * Expõe a instância oficial `app = ServerApp(server_fn=server_fn)`.
* **[`simulacao_direta.py`](file:///c:/Users/jsous/Desktop/flower_puro/simulacao_direta.py)**:
  * Executor de linha de comando autônomo. Permite testar o aprendizado federado diretamente na máquina local sem precisar inicializar daemons ou orquestradores de multiprocesso (Ray/SuperLink).
  * Gera checkpoints `.pt`, gráficos PNG (`curvas_convergencia.png`), arquivo JSON (`metricas_federadas.json`) e tabela de resultados em Markdown (`TABELA_RESULTADOS.md`).
* **[`executar_flower_puro.ipynb`](file:///c:/Users/jsous/Desktop/flower_puro/executar_flower_puro.ipynb)**:
  * Notebook interativo desenhado especificamente para o Google Colab com aceleração por GPU T4 gratuita.
  * Possui formulário gráfico interativo (`#@title`, `#@param`) para rodar simulações sem alterar código, sistema de auto-recuperação de dependências e download nativo dos resultados em arquivo `.zip`.

---

## 🔄 Fluxo de Dados e Conversão de Tensores

O Flower é agnóstico a frameworks de deep learning. Para alcançar essa interoperabilidade, a comunicação ocorre via **arrays NumPy**:

```text
[PyTorch: state_dict()]  ──(get_weights)──>  [List[np.ndarray]]  ──(Rede Flower)──>  Servidor
[PyTorch: state_dict()]  <──(set_weights)──  [List[np.ndarray]]  <──(Rede Flower)──  Servidor
```

* **Serialização (`get_weights`)**:
  Extrai os tensores de cada camada do modelo PyTorch, move-os para a CPU e converte-os em arrays NumPy:
  ```python
  def get_weights(net: nn.Module) -> list[np.ndarray]:
      return [val.cpu().numpy() for _, val in net.state_dict().items()]
  ```
* **Desserialização (`set_weights`)**:
  Recebe a lista de matrizes NumPy agregadas pelo servidor, reconstrói os tensores PyTorch e carrega no modelo via `load_state_dict`:
  ```python
  def set_weights(net: nn.Module, parameters: list[np.ndarray]) -> None:
      params_dict = zip(net.state_dict().keys(), parameters)
      state_dict = {k: torch.tensor(v) for k, v in params_dict}
      net.load_state_dict(state_dict, strict=True)
  ```

---

## 📊 Estratégias de Agregação e Particionamento

### 1. Estratégias de Agregação Suportadas

1. **`FedAvg` (Federated Averaging - McMahan et al., 2017)**:
   Média ponderada clássica calculada sobre o volume de dados de cada nó cliente:
   $$\mathbf{W}_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{W}_{t+1}^k$$
2. **`FedProx` (Li et al., 2020)**:
   Adiciona um termo proximal $\frac{\mu}{2} \|\mathbf{w} - \mathbf{w}^t\|^2$ à função de perda local, limitando o distanciamento dos modelos locais em relação ao modelo global sob alta heterogeneidade estatística.
3. **`FedMedian` (Yin et al., 2018)**:
   Calcula a mediana coordenada por coordenada das matrizes de pesos de todos os clientes. Fornece robustez contra nós ruidosos ou valores discrepantes (*outliers*).

### 2. Particionamento dos Dados (`flwr-datasets`)

* **IID (`IidPartitioner`)**: Os dados do CIFAR-10 são divididos de maneira balanceada e uniforme. Cada cliente recebe aproximadamente o mesmo número de imagens de cada uma das 10 classes.
* **Não-IID Dirichlet (`DirichletPartitioner`)**: Modela o comportamento real onde cada nó possui preferências ou perfis distintos de dados, amostrados a partir de uma distribuição $\text{Dirichlet}(\alpha)$:
  * **$\alpha \ge 100.0$**: Distribuição praticamente idêntica a IID (homogênea).
  * **$\alpha = 1.0$**: Assimetria suave entre os nós.
  * **$\alpha = 0.5$**: Assimetria moderada (clientes concentram dados em poucas classes).
  * **$\alpha = 0.1$**: Assimetria severa (*Névoa de Guerra* federada — cada cliente detém quase exclusivamente 1 ou 2 classes).

---

## ☁️ Como o Projeto Roda no Google Colab

O notebook [`executar_flower_puro.ipynb`](file:///c:/Users/jsous/Desktop/flower_puro/executar_flower_puro.ipynb) conecta-se diretamente ao repositório GitHub [**JM3L0/flower_puro**](https://github.com/JM3L0/flower_puro), garantindo que o Colab sempre execute o código mais recente do projeto:

### Passo a Passo de Execução no Colab

```text
[1. Checagem GPU T4] ──> [2. Git Clone/Pull] ──> [3. Formulário & Treino] ──> [4. Gráficos & Tabela] ──> [5. Download .ZIP]
```

1. **Célula 1 - Verificação de GPU**: Executa `!nvidia-smi` para confirmar a alocação de uma GPU NVIDIA (como a T4 gratuita).
2. **Célula 2 - Sincronização com o GitHub e Instalação**:
   * Clona o repositório `JM3L0/flower_puro` na VM do Colab (ou executa `git pull` se a pasta já existir).
   * Instala as dependências de `requirements.txt` e o pacote em modo editável (`pip install -e .`).
3. **Célula 3 - Formulário Interativo & Simulação**:
   * Permite configurar as rodadas, clientes, taxa de aprendizado e heterogeneidade Dirichlet via formulário GUI (`#@title`, `#@param`).
   * **Executa diretamente os módulos do repositório**: Invoca `simulacao_direta.executar_simulacao()`, que utiliza os arquivos do pacote [`flower_puro`](file:///c:/Users/jsous/Desktop/flower_puro/flower_puro).
   * **Mecanismo de Auto-Recuperação Defensivo**: Se a Célula 2 for pulada, a Célula 3 clona o repositório e instala as dependências automaticamente antes de rodar.
4. **Célula 4 - Visualização Gráfica e Tabela**: Exibe em tela a imagem `curvas_convergencia.png` e renderiza a tabela Markdown `TABELA_RESULTADOS.md` com as métricas consolidadas de cada rodada.
5. **Célula 5 - Download Nativo dos Resultados**: Compacta o modelo salvo (`modelo_federado.pt`), métricas JSON e gráficos em `resultados_flower_puro.zip` e dispara o download direto para o navegador.

> [!TIP]
> **Como testar alterações feitas no computador no Colab:**
> 1. Altere os arquivos `.py` no seu computador (ex: `flower_puro/task.py`);
> 2. Envie as alterações para o GitHub com `git commit` e `git push origin main`;
> 3. No Colab, execute novamente a Célula 2 (ou dê `Ctrl + F9`); o `git pull` baixará suas novidades automaticamente!

---

## 🛠️ Guia de Alterações no Código

Este guia ensina exatamente onde e como modificar o projeto, tanto no código modular do repositório quanto dentro do notebook do Google Colab.

---

### 1. Alterando a Arquitetura da Rede Neural

#### A. No Repositório Local
Abra o arquivo [`flower_puro/task.py`](file:///c:/Users/jsous/Desktop/flower_puro/flower_puro/task.py#L33-L56). A classe `Net` define a arquitetura convolucional.

**Exemplo: Adicionando Dropout e Batch Normalization:**
```python
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 32 canais x 8 x 8 após duas reduções de pooling
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
```

#### B. No Google Colab
No arquivo `executar_flower_puro.ipynb`, localize a **Célula 3** ("⚙️ Painel de Controle da Simulação Federada").
A definição da classe `class Net(nn.Module):` está implementada logo abaixo das variáveis do formulário. Basta editar a arquitetura diretamente nessa célula e rodá-la novamente.

---

### 2. Alterando Otimizador e Hiperparâmetros

#### A. No Repositório Local
* **Otimizador Local (ex: Trocar SGD por AdamW)**:
  No arquivo [`flower_puro/task.py`](file:///c:/Users/jsous/Desktop/flower_puro/flower_puro/task.py#L170-L172), altere a função `train`:
  ```python
  # De:
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

  # Para:
  optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
  ```
* **Hiperparâmetros Padrão do Flower CLI**:
  No arquivo [`pyproject.toml`](file:///c:/Users/jsous/Desktop/flower_puro/pyproject.toml#L28-L39), você pode ajustar os valores padrões usados pelo comando `flwr run .`:
  ```toml
  [tool.flwr.app.config]
  num-server-rounds = 15      # Aumenta de 10 para 15 rodadas
  local-epochs = 3           # Aumenta de 2 para 3 épocas locais
  learning-rate = 0.001       # Ajusta a taxa de aprendizado
  batch-size = 32            # Reduz o tamanho do lote
  dirichlet-alpha = 0.5      # Ativa assimetria Não-IID moderada
  ```

#### B. No Google Colab
* Você pode alterar os hiperparâmetros diretamente pelo **Formulário Visual** da Célula 3 (selecionando rodadas, clientes, épocas locais, learning rate e distribuição).
* Se desejar alterar o otimizador no Colab, desça na Célula 3 até a função `train_client` e substitua a linha do `torch.optim.SGD` pelo otimizador de sua preferência.

---

### 3. Alterando a Estratégia de Agregação

#### A. No Repositório Local
Abra o arquivo [`flower_puro/server_app.py`](file:///c:/Users/jsous/Desktop/flower_puro/flower_puro/server_app.py#L82-L120). O Flower disponibiliza várias estratégias prontas que podem ser importadas de `flwr.server.strategy`, tais como:
* `FedAvg` (Média ponderada padrão)
* `FedProx` (Termo proximal para redes heterogêneas)
* `FedMedian` (Mediana coordenada)
* `FedTrimmedAvg` (Média aparada contra envenenamento de pesos)
* `FedAvgM` (FedAvg com Momentum no servidor)

**Exemplo: Ativando `FedTrimmedAvg` em `server_app.py`:**
```python
from flwr.server.strategy import FedTrimmedAvg

# Na função server_fn:
strategy = FedTrimmedAvg(
    fraction_fit=fraction_fit,
    fraction_evaluate=fraction_evaluate,
    min_fit_clients=10,
    min_available_clients=10,
    evaluate_fn=evaluate_fn,
    beta=0.2, # Descarta 20% das atualizações mais extremas de cada coordenada
)
```

Na simulação direta via terminal ([`simulacao_direta.py`](file:///c:/Users/jsous/Desktop/flower_puro/simulacao_direta.py)), basta passar a flag `--strategy`:
```bash
python simulacao_direta.py --strategy FedMedian --rounds 15 --clients 10
```

#### B. No Google Colab
No formulário da Célula 3 do notebook, há um seletor visual:
```python
estrategia_agregacao = "FedAvg" #@param ["FedAvg", "FedMedian"]
```
Você pode adicionar novas estratégias no formulário ou customizar o laço de agregação no bloco `if estrategia_agregacao == ...`.

---

### 4. Trocando o Dataset

Por padrão, o projeto utiliza o dataset `uoft-cs/cifar10` via biblioteca `flwr-datasets`.

#### A. No Repositório Local
No arquivo [`flower_puro/task.py`](file:///c:/Users/jsous/Desktop/flower_puro/flower_puro/task.py#L129-L155):
1. **Alterar o identificador do dataset**:
   Substitua `"uoft-cs/cifar10"` por outro dataset do Hugging Face / Flower Datasets (como `"zalando-datasets/fashion_mnist"` ou `"ylecun/mnist"`):
   ```python
   _cached_fds = FederatedDataset(dataset="zalando-datasets/fashion_mnist", partitioners={"train": partitioner})
   ```
2. **Ajustar as dimensões da rede**:
   Se mudar para um dataset em escala de cinza (1 canal) como MNIST ($28 \times 28 \times 1$), ajuste o canal de entrada em `Net`:
   ```python
   self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
   ```
   E altere a normalização em `_transforms` para apenas 1 canal:
   ```python
   _transforms = Compose([
       ToTensor(),
       Normalize((0.5,), (0.5,)),
   ])
   ```

#### B. No Google Colab
A mesma lógica é aplicável na Célula 3 de `executar_flower_puro.ipynb`, editando a linha `FederatedDataset(dataset="uoft-cs/cifar10", ...)` e os canais da classe `Net`.

---

### 5. Sincronizando Alterações: Colab $\leftrightarrow$ Git

Se você fez modificações diretamente no ambiente do Google Colab e deseja salvá-las no repositório:

1. **Baixar o Notebook alterado do Colab**:
   No Colab, vá em **Arquivo $\to$ Fazer download $\to$ Fazer download do .ipynb** e salve o arquivo substituindo o [executar_flower_puro.ipynb](file:///c:/Users/jsous/Desktop/flower_puro/executar_flower_puro.ipynb) local.
2. **Salvar e Enviar via Git**:
   No terminal da pasta do seu projeto:
   ```bash
   git add executar_flower_puro.ipynb
   git commit -m "feat: atualiza notebook do Colab com novos experimentos"
   git push origin main
   ```
3. **Ou salvar diretamente no GitHub via Colab**:
   No menu do Colab, clique em **Arquivo $\to$ Salvar uma cópia no GitHub**, selecione o repositório `JM3L0/flower_puro` e confirme o commit.

---

## 🚀 Modos de Execução Disponíveis

| Modo | Arquivo / Comando | Onde Executa | Aceleração | Indicado Para |
|---|---|---|:---:|---|
| **Google Colab** | `executar_flower_puro.ipynb` | Nuvem (Google Colab) | GPU T4 Gratuita | Experimentos imediatos, sem instalar nada na máquina local |
| **Simulação Direta** | `simulacao_direta.py` | Máquina Local / Terminal | GPU Local ou CPU | Prototipagem rápida, scripts automatizados e geração de gráficos |
| **Flower CLI 1.x** | `flwr run .` | Máquina Local / SuperLink | Multiprocesso Ray | Padrão oficial de produção do ecossistema Flower |

---

### Execução Local Rápida (Exemplos)

```bash
# 1. Instalar as dependências do projeto
pip install -r requirements.txt

# 2. Executar simulação homogênea padrão (IID - FedAvg)
python simulacao_direta.py --strategy FedAvg --rounds 10 --clients 10 --alpha 100.0

# 3. Executar simulação sob assimetria Não-IID severa (Dirichlet α=0.5 com FedProx)
python simulacao_direta.py --strategy FedProx --rounds 10 --clients 10 --alpha 0.5
```

Ao término, os arquivos serão salvos na pasta `resultados/`:
* `curvas_convergencia.png`: Gráficos de acurácia global e perda (*loss*).
* `modelo_federado.pt`: Checkpoint PyTorch com os pesos do modelo global treinado.
* `TABELA_RESULTADOS.md`: Resumo tabular das métricas por rodada.
* `metricas_federadas.json`: Métricas estruturadas para análise estatística.
