"""simulacao_direta.py — Executor Direto e Autocontido de Aprendizado Federado Canônico.

Permite rodar a simulação do Flower diretamente via linha de comando ou no Colab,
sem necessidade de daemons ou orquestradores externos.

Uso:
    python simulacao_direta.py --strategy FedAvg --rounds 10 --clients 10 --alpha 100.0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

# Garante suporte a UTF-8 nos logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import torch

from flower_puro.task import (
    Net,
    get_weights,
    load_centralized_test_data,
    load_data,
    set_weights,
    test,
    train,
)


def federated_average(client_weights: list[list[np.ndarray]], client_sizes: list[int]) -> list[np.ndarray]:
    """FedAvg Canônico (McMahan et al., 2017): Média ponderada pelo número de amostras."""
    total_samples = sum(client_sizes)
    num_layers = len(client_weights[0])
    aggregated = []

    for layer_idx in range(num_layers):
        layer_sum = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for client_idx in range(len(client_weights)):
            weight = client_sizes[client_idx] / total_samples
            layer_sum += weight * client_weights[client_idx][layer_idx].astype(np.float64)
        aggregated.append(layer_sum.astype(client_weights[0][layer_idx].dtype))
    return aggregated


def federated_median(client_weights: list[list[np.ndarray]]) -> list[np.ndarray]:
    """FedMedian: Mediana coordenada por coordenada (robusta a variações extremas)."""
    num_layers = len(client_weights[0])
    aggregated = []

    for layer_idx in range(num_layers):
        stacked = np.stack([client_weights[c][layer_idx] for c in range(len(client_weights))], axis=0)
        med = np.median(stacked, axis=0)
        aggregated.append(med.astype(client_weights[0][layer_idx].dtype))
    return aggregated


def plotar_convergencia(historico: list[dict], output_dir: str, strategy: str) -> str:
    """Gera o gráfico de convergência de Acurácia e Loss ao longo das rodadas."""
    os.makedirs(output_dir, exist_ok=True)
    rounds = [h["round"] for h in historico]
    accs = [h["test_accuracy"] * 100 for h in historico]
    losses = [h["test_loss"] for h in historico]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Gráfico 1: Acurácia Global
    ax1.plot(rounds, accs, marker="o", color="#1976D2", linewidth=2.2, label=f"{strategy} (Global)")
    ax1.set_title("Evolução da Acurácia Global de Teste", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Rodada Federada", fontsize=11)
    ax1.set_ylabel("Acurácia (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="lower right")

    # Gráfico 2: Perda (Loss)
    ax2.plot(rounds, losses, marker="s", color="#D32F2F", linewidth=2.2, label=f"{strategy} (Loss)")
    ax2.set_title("Evolução da Perda (Cross-Entropy Loss)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Rodada Federada", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right")

    fig.suptitle(f"Curvas de Aprendizado Federado ({strategy} - CIFAR-10)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    caminho_grafico = os.path.join(output_dir, "curvas_convergencia.png")
    fig.savefig(caminho_grafico, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return caminho_grafico


def executar_simulacao(
    strategy: str = "FedAvg",
    num_rounds: int = 10,
    num_clients: int = 10,
    local_epochs: int = 2,
    learning_rate: float = 0.01,
    batch_size: int = 64,
    dirichlet_alpha: float = 100.0,
    seed: int = 42,
    output_dir: str = "resultados",
) -> dict:
    """Executa o ciclo completo de treinamento federado canônico."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    dist_nome = "IID (Homogênea)" if dirichlet_alpha >= 100.0 else f"Não-IID (Dirichlet α={dirichlet_alpha})"

    print("=" * 75)
    print(f"  🌸 SIMULAÇÃO FEDERADA DIDÁTICA ({strategy})")
    print("=" * 75)
    print(f"  • Estratégia de Agregação : {strategy}")
    print(f"  • Rodadas Globais         : {num_rounds}")
    print(f"  • Total de Clientes       : {num_clients}")
    print(f"  • Épocas Locais           : {local_epochs}")
    print(f"  • Taxa de Aprendizado (LR): {learning_rate}")
    print(f"  • Batch Size Local        : {batch_size}")
    print(f"  • Distribuição de Dados   : {dist_nome}")
    print(f"  • Dispositivo de Cálculo  : {device}")
    print(f"  • Semente (Seed)          : {seed}")
    print("=" * 75 + "\n")

    # 1. Inicializa modelo global e carrega teste global
    print("[1/3] Carregando conjunto de teste global (10.000 imagens)...")
    testloader = load_centralized_test_data(batch_size=128)
    global_model = Net().to(device)

    # Avaliação inicial (rodada 0)
    init_loss, init_acc = test(global_model, testloader, device)
    print(f"  ▶ Baseline Inicial (Sem treino): Acurácia = {init_acc * 100:.2f}% | Loss = {init_loss:.4f}\n")

    historico = []
    tempo_inicio = time.time()

    # 2. Loop de Rodadas Federadas
    print(f"[2/3] Iniciando {num_rounds} rodadas de Aprendizado Federado...")
    for round_idx in range(1, num_rounds + 1):
        round_start = time.time()
        client_weights = []
        client_sizes = []
        train_losses = []

        global_weights = get_weights(global_model)

        # Treinamento local em cada cliente
        for client_id in range(num_clients):
            trainloader, _ = load_data(
                partition_id=client_id,
                num_partitions=num_clients,
                batch_size=batch_size,
                dirichlet_alpha=dirichlet_alpha,
            )

            local_model = Net().to(device)
            set_weights(local_model, global_weights)

            c_loss, c_acc = train(
                net=local_model,
                trainloader=trainloader,
                epochs=local_epochs,
                lr=learning_rate,
                device=device,
            )

            client_weights.append(get_weights(local_model))
            client_sizes.append(len(trainloader.dataset))
            train_losses.append(c_loss)

        # Agregação central no servidor
        if strategy == "FedMedian":
            aggregated_weights = federated_median(client_weights)
        else:  # FedAvg (ou FedProx)
            aggregated_weights = federated_average(client_weights, client_sizes)

        set_weights(global_model, aggregated_weights)

        # Avaliação global no servidor
        round_loss, round_acc = test(global_model, testloader, device)
        round_time = time.time() - round_start

        historico.append({
            "round": round_idx,
            "test_accuracy": float(round_acc),
            "test_loss": float(round_loss),
            "avg_train_loss": float(np.mean(train_losses)),
            "round_time_s": float(round_time),
        })

        print(
            f"  [Rodada {round_idx:02d}/{num_rounds:02d}] "
            f"Acurácia Global: {round_acc * 100:6.2f}% | "
            f"Loss: {round_loss:.4f} | "
            f"Tempo: {round_time:.1f}s"
        )

    tempo_total = time.time() - tempo_inicio

    # 3. Salva resultados e gráficos
    print(f"\n[3/3] Treinamento finalizado em {tempo_total:.1f}s! Salvando resultados...")

    # Salva pesos finais do modelo
    caminho_modelo = os.path.join(output_dir, "modelo_federado.pt")
    torch.save(global_model.state_dict(), caminho_modelo)

    # Gera figura de convergência
    caminho_figura = plotar_convergencia(historico, output_dir, strategy)

    # Salva JSON de métricas
    resumo = {
        "config": {
            "strategy": strategy,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dirichlet_alpha": dirichlet_alpha,
            "seed": seed,
            "device": str(device),
        },
        "final_accuracy": historico[-1]["test_accuracy"],
        "final_loss": historico[-1]["test_loss"],
        "total_time_seconds": tempo_total,
        "rounds": historico,
    }

    caminho_json = os.path.join(output_dir, "metricas_federadas.json")
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2)

    # Salva resumo em Markdown
    caminho_md = os.path.join(output_dir, "TABELA_RESULTADOS.md")
    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(f"# 🌸 Resultados da Simulação Federada ({strategy})\n\n")
        f.write(f"- **Acurácia Global Final**: **{resumo['final_accuracy'] * 100:.2f}%**\n")
        f.write(f"- **Perda (Loss) Final**: {resumo['final_loss']:.4f}\n")
        f.write(f"- **Tempo Total de Execução**: {tempo_total:.1f} segundos\n")
        f.write(f"- **Distribuição dos Dados**: {dist_nome}\n\n")
        f.write("| Rodada | Acurácia Global (%) | Loss Teste | Loss Treino Médio | Duração (s) |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|\n")
        for h in historico:
            f.write(f"| {h['round']} | {h['test_accuracy'] * 100:.2f}% | {h['test_loss']:.4f} | {h['avg_train_loss']:.4f} | {h['round_time_s']:.1f} |\n")

    print("\n" + "=" * 75)
    print(f"  ✔ SUCESSO! Acurácia Final: {resumo['final_accuracy'] * 100:.2f}%")
    print(f"  📁 Modelo Salvo    : {caminho_modelo}")
    print(f"  📈 Gráfico Gerado  : {caminho_figura}")
    print(f"  📋 Tabela Resumo   : {caminho_md}")
    print(f"  📊 Métricas JSON   : {caminho_json}")
    print("=" * 75 + "\n")

    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulação Didática de Aprendizado Federado Canônico.")
    parser.add_argument("--strategy", type=str, default="FedAvg", choices=["FedAvg", "FedMedian", "FedProx"],
                        help="Estratégia de agregação do servidor")
    parser.add_argument("--rounds", type=int, default=10, help="Número de rodadas federadas globais")
    parser.add_argument("--clients", type=int, default=10, help="Quantidade de clientes na federação")
    parser.add_argument("--epochs", type=int, default=2, help="Número de épocas de treino local por rodada")
    parser.add_argument("--lr", type=float, default=0.01, help="Taxa de aprendizado local (SGD)")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamanho do lote de treino local")
    parser.add_argument("--alpha", type=float, default=100.0,
                        help="Heterogeneidade Dirichlet (>=100: IID homogêneo; <=1.0: Não-IID)")
    parser.add_argument("--seed", type=int, default=42, help="Semente de aleatoriedade")
    parser.add_argument("--output_dir", type=str, default="resultados", help="Diretório de saída")

    args = parser.parse_args()

    executar_simulacao(
        strategy=args.strategy,
        num_rounds=args.rounds,
        num_clients=args.clients,
        local_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        dirichlet_alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
