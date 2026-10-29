# src/model_scripts/check_metrics.py
"""
Quality gate для деплоя модели фрод-детекции.

Пороги выставлены по метрикам актуальной Model v2
(Accuracy=0.9345, F1=0.7783, Precision_fraud=0.6745,
Recall_fraud=0.92, ROC-AUC=0.97818) с небольшим допуском
на естественный разброс между запусками.

Логика:
- Для recall_fraud и roc_auc (ключевые для безопасности)
  порог почти на уровне v2 — не допускаем деградации.
- Для f1_score и precision_fraud — небольшой допуск (0.01),
  т.к. эти метрики более шумные при переобучении.
- Accuracy не используем как отдельный gate: при сильном
  дисбалансе классов она малоинформативна (0.93 достигается
  тривиальным предсказанием "всё нормально").
"""
import os
import sys
from pathlib import Path

import mlflow

# ==========================================
# ПОРОГИ (обоснованы метриками Model v2)
# ==========================================
THRESHOLDS = {
    # Ключевая метрика безопасности: не пропускать фрод.
    # v2 = 0.92 → порог 0.90 (допуск 0.02 на разброс).
    "recall_fraud": 0.90,

    # Качество ранжирования транзакций по риску.
    # v2 = 0.97818 → порог 0.975 (допуск ~0.003).
    "roc_auc": 0.975,

    # Баланс precision/recall. v2 = 0.7783 → порог 0.76.
    "f1_score": 0.76,

    # Снижение ложных блокировок легитимных клиентов.
    # v2 = 0.6745 → порог 0.66 (допуск ~0.015).
    "precision_fraud": 0.66,
}


def main():
    project_root = Path(__file__).parent.parent.parent
    mlflow_dir = project_root / "mlruns"

    if not mlflow_dir.exists():
        print(f"❌ MLflow directory not found at {mlflow_dir}")
        sys.exit(1)

    mlflow.set_tracking_uri(mlflow_dir.as_uri())

    experiment_name = "fraud_detection"  # при необходимости поправьте
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found")
        sys.exit(1)

    # Берём последний run эксперимента
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        print("❌ No runs found")
        sys.exit(1)

    latest_run = runs.iloc[0]
    run_id = latest_run["run_id"]
    metrics = latest_run.filter(like="metrics.").to_dict()

    print(f"📊 Checking metrics for run: {run_id}")

    failed = False
    summary_lines = [
        "## Model Quality Gate",
        f"**Run ID:** `{run_id}`",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
    ]

    for metric_name, threshold in THRESHOLDS.items():
        value = metrics.get(f"metrics.{metric_name}")
        if value is None:
            status = "❌ MISSING"
            failed = True
            val_str = "N/A"
        else:
            val_str = f"{value:.4f}"
            status = "✅ PASS" if value >= threshold else "❌ FAIL"
            if value < threshold:
                failed = True

        summary_lines.append(
            f"| {metric_name} | {val_str} | ≥ {threshold} | {status} |"
        )

    # Пишем в GitHub Job Summary (видно на странице Approve)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n".join(summary_lines))
            f.write("\n")

    print("\n".join(summary_lines))

    if failed:
        print("\n❌ Quality gate failed. Pipeline stops before approval.")
        sys.exit(1)

    print("\n✅ All metrics passed. Waiting for manual approval.")


if __name__ == "__main__":
    main()