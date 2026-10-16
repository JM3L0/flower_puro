"""client_app.py — Implementação Canônica do Cliente Federado (ClientApp).

Este módulo define o comportamento de cada nó cliente na federação:
1. Recebe os parâmetros globais enviados pelo servidor (`fit`).
2. Treina localmente por N épocas usando apenas seus dados privados.
3. Devolve os parâmetros atualizados e as métricas para a agregação global.
4. Avalia opcionalmente o modelo em seus dados de validação locais (`evaluate`).
"""

from __future__ import annotations

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from flower_puro.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    """Cliente federado canônico baseado na interface NumPyClient do Flower."""

    def __init__(
        self,
        partition_id: int,
        net: Net,
        trainloader,
        valloader,
        local_epochs: int,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device

    def get_parameters(self, config: dict[str, str]) -> list:
        """Retorna os pesos locais atuais do modelo."""
        return get_weights(self.net)

    def fit(self, parameters: list, config: dict[str, str]) -> tuple[list, int, dict]:
        """Recebe parâmetros globais do servidor, treina localmente e devolve pesos atualizados."""
        # 1. Atualiza pesos locais com o modelo global recebido do servidor
        set_weights(self.net, parameters)

        # 2. Permite que o servidor sobrescreva épocas ou learning rate na rodada atual
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("learning_rate", self.learning_rate))

        # 3. Executa o treinamento local
        train_loss, train_acc = train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=epochs,
            lr=lr,
            device=self.device,
        )

        num_examples = len(self.trainloader.dataset)
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "client_id": self.partition_id,
        }
        return get_weights(self.net), num_examples, metrics

    def evaluate(self, parameters: list, config: dict[str, str]) -> tuple[float, int, dict]:
        """Avalia os parâmetros globais no conjunto de validação local do cliente."""
        set_weights(self.net, parameters)
        val_loss, val_acc = test(net=self.net, testloader=self.valloader, device=self.device)
        num_examples = len(self.valloader.dataset)
        return float(val_loss), num_examples, {"accuracy": float(val_acc)}


def client_fn(context: Context) -> NumPyClient:
    """Fábrica de instâncias de clientes federados chamada pelo motor de simulação."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config.get("num-partitions", 10)

    # Hiperparâmetros configurados no pyproject.toml ou passados na execução
    cfg = context.run_config
    batch_size = int(cfg.get("batch-size", 64))
    local_epochs = int(cfg.get("local-epochs", 2))
    learning_rate = float(cfg.get("learning-rate", 0.01))
    dirichlet_alpha = float(cfg.get("dirichlet-alpha", 100.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    trainloader, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        dirichlet_alpha=dirichlet_alpha,
    )

    return FlowerClient(
        partition_id=partition_id,
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    ).to_client()


# Ponto de entrada do ClientApp para o comando `flwr run`
app = ClientApp(client_fn=client_fn)
