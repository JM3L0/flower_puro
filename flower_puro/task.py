"""task.py — Definição do Modelo PyTorch, Carregamento de Dados e Rotinas de Treino.

Este módulo é o coração didático do aprendizado de máquina:
1. Define uma arquitetura de Rede Neural Convolucional (CNN) simples para CIFAR-10.
2. Configura a partição federada dos dados (IID ou Não-IID via Dirichlet).
3. Provê funções canônicas de treino local (`train`) e avaliação (`test`).
4. Converte pesos PyTorch para arrays NumPy (padrão de comunicação do Flower).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset  # type: ignore
from flwr_datasets import FederatedDataset  # type: ignore
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner  # type: ignore
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor  # type: ignore

# 10 classes canônicas do dataset CIFAR-10
CIFAR10_CLASSES = [
    "Avião", "Automóvel", "Pássaro", "Gato", "Cervo",
    "Cachorro", "Sapo", "Cavalo", "Navio", "Caminhão"
]


# =============================================================================
# 1. ARQUITETURA DA REDE NEURAL (CNN)
# =============================================================================

class Net(nn.Module):
    """Rede Convolucional simples (CNN) adaptada para CIFAR-10 (imagens 32x32x3)."""

    def __init__(self) -> None:
        super().__init__()
        # Bloco convolucional 1: 3 canais de entrada (RGB) -> 6 filtros 5x5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bloco convolucional 2: 6 canais -> 16 filtros 5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # Camadas densas totalmente conectadas
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagação direta (Forward Pass)."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# 2. CONVERSÃO DE PESOS ENTRE PYTORCH E NUMPY (FLOWER)
# =============================================================================

def get_weights(net: nn.Module) -> list[np.ndarray]:
    """Extrai os parâmetros do modelo PyTorch como uma lista de matrizes NumPy."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: list[np.ndarray]) -> None:
    """Carrega os pesos agregados pelo servidor de volta na rede PyTorch."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)


# =============================================================================
# 3. CARREGAMENTO E PARTIÇÃO FEDERADA DOS DADOS
# =============================================================================

# Transformações de pré-processamento de imagem
_transforms = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def _apply_transforms(batch: dict) -> dict:
    """Aplica conversão para tensor e normalização em lote."""
    batch["img"] = [_transforms(img) for img in batch["img"]]
    return batch


# Cache para evitar recarregar o dataset da Hugging Face a cada partição
_cached_fds: FederatedDataset | None = None
_cached_alpha: float | None = None
_cached_partitions: int | None = None


def load_data(
    partition_id: int,
    num_partitions: int = 10,
    batch_size: int = 64,
    dirichlet_alpha: float = 100.0,
) -> tuple[DataLoader, DataLoader]:
    """Carrega o subconjunto local de treino e teste de um cliente federado específico.

    Args:
        partition_id: ID do cliente (0 a num_partitions-1).
        num_partitions: Quantidade total de clientes simulados.
        batch_size: Tamanho do minilote para DataLoader.
        dirichlet_alpha: Grau de heterogeneidade dos dados:
            - alpha >= 100.0: Distribuição IID (classes balanceadas igualmente entre clientes).
            - alpha <= 1.0: Distribuição Não-IID (heterogênea / assimétrica).

    Returns:
        (trainloader, valloader): DataLoaders PyTorch prontos para treinamento e validação.
    """
    global _cached_fds, _cached_alpha, _cached_partitions

    if _cached_fds is None or _cached_alpha != dirichlet_alpha or _cached_partitions != num_partitions:
        if dirichlet_alpha >= 100.0:
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=dirichlet_alpha,
                min_partition_size=10,
                self_balancing=True,
            )
        _cached_fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner})
        _cached_alpha = dirichlet_alpha
        _cached_partitions = num_partitions

    partition = _cached_fds.load_partition(partition_id, "train")
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(_apply_transforms)

    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    valloader = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
        shuffle=False,
    )
    return trainloader, valloader


def load_centralized_test_data(batch_size: int = 128) -> DataLoader:
    """Carrega o conjunto de teste global completo (10.000 imagens) para avaliação no servidor."""
    test_ds = load_dataset("uoft-cs/cifar10", split="test")
    test_ds = test_ds.with_transform(_apply_transforms)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# =============================================================================
# 4. ROTINAS DE TREINAMENTO LOCAL E AVALIAÇÃO
# =============================================================================

def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[float, float]:
    """Treina o modelo nos dados locais do cliente por um número fixo de épocas."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    net.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Avalia o modelo em um conjunto de dados e calcula a Perda (Loss) e Acurácia."""
    criterion = nn.CrossEntropyLoss()
    net.eval()
    net.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
