"""server_app.py — Implementação Canônica do Servidor Federado (ServerApp).

Este módulo define a orquestração central da federação:
1. Inicializa os pesos do modelo global.
2. Seleciona os clientes participantes em cada rodada.
3. Agrega as atualizações locais usando FedAvg (ou FedProx/FedMedian).
4. Avalia o modelo global contra o conjunto de teste centralizado do CIFAR-10.
"""

from __future__ import annotations

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedMedian, FedProx

from flower_puro.task import (
    Net,
    get_weights,
    load_centralized_test_data,
    set_weights,
    test,
)


def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Calcula a média ponderada das métricas enviadas pelos clientes federados."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}
    for num_examples, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + (v * num_examples)

    for k in aggregated:
        aggregated[k] /= total_examples
    return aggregated


def get_evaluate_fn(testloader, device: torch.device):
    """Retorna uma função de avaliação centralizada chamada pelo servidor ao fim de cada rodada."""

    def evaluate(server_round: int, parameters: list, config: dict):
        net = Net().to(device)
        set_weights(net, parameters)
        loss, accuracy = test(net=net, testloader=testloader, device=device)
        print(f"\n[Servidor - Rodada {server_round}] Acurácia Global de Teste: {accuracy * 100:.2f}% | Loss: {loss:.4f}")
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate


def server_fn(context: Context) -> ServerAppComponents:
    """Configura e instancia os componentes do servidor federado."""
    cfg = context.run_config

    num_rounds = int(cfg.get("num-server-rounds", 10))
    fraction_fit = float(cfg.get("fraction-fit", 1.0))
    fraction_evaluate = float(cfg.get("fraction-evaluate", 1.0))
    strategy_name = str(cfg.get("strategy", "FedAvg"))
    proximal_mu = float(cfg.get("proximal-mu", 1.0))

    # Inicializa pesos globais
    initial_net = Net()
    initial_parameters = ndarrays_to_parameters(get_weights(initial_net))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testloader = load_centralized_test_data(batch_size=128)
    evaluate_fn = get_evaluate_fn(testloader=testloader, device=device)

    # Configuração de hiperparâmetros enviada aos clientes a cada rodada
    def fit_config(server_round: int) -> dict:
        return {
            "server_round": server_round,
            "local_epochs": int(cfg.get("local-epochs", 2)),
            "learning_rate": float(cfg.get("learning-rate", 0.01)),
        }

    # Seleção da estratégia canônica do Flower
    if strategy_name.lower() == "fedprox":
        strategy = FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=int(cfg.get("min-fit-clients", 10)),
            min_evaluate_clients=int(cfg.get("min-evaluate-clients", 10)),
            min_available_clients=int(cfg.get("min-available-clients", 10)),
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
            proximal_mu=proximal_mu,
        )
    elif strategy_name.lower() == "fedmedian":
        strategy = FedMedian(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=int(cfg.get("min-fit-clients", 10)),
            min_evaluate_clients=int(cfg.get("min-evaluate-clients", 10)),
            min_available_clients=int(cfg.get("min-available-clients", 10)),
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
        )
    else:  # Padrão: FedAvg
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=int(cfg.get("min-fit-clients", 10)),
            min_evaluate_clients=int(cfg.get("min-evaluate-clients", 10)),
            min_available_clients=int(cfg.get("min-available-clients", 10)),
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=weighted_average,
            initial_parameters=initial_parameters,
        )

    server_config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=server_config)


# Ponto de entrada do ServerApp para o comando `flwr run`
app = ServerApp(server_fn=server_fn)
