# FinStreamAI Helm Chart

Production LLMOps fraud detection stack — one-command install.

## Install
```bash
helm repo add finstreamai https://deepika1715.github.io/FinStreamAI
helm repo update
helm install my-finstreamai finstreamai/finstreamai
```

## Install with custom values
```bash
helm install my-finstreamai finstreamai/finstreamai \
  --set fraudApi.model.threshold=0.8 \
  --set llmops.enabled=true \
  --set monitoring.grafana.enabled=true
```

## Uninstall
```bash
helm uninstall my-finstreamai
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fraudApi.replicaCount` | `2` | Number of API replicas |
| `fraudApi.model.threshold` | `0.5` | Fraud score cutoff |
| `fraudApi.model.version` | `xgboost-v1` | Model version tag |
| `kafka.enabled` | `true` | Enable Kafka stream |
| `monitoring.enabled` | `true` | Enable Prometheus + Grafana |
| `llmops.enabled` | `true` | Enable RAG explanation layer |
| `llmops.promptVersion` | `explain_v1` | Prompt template version |
| `demo.enabled` | `true` | Enable Streamlit demo UI |
| `mlflow.enabled` | `true` | Enable MLflow tracking |
