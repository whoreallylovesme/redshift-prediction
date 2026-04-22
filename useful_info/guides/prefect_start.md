# Prefect: quick start

## Запуск flow без дашборда

```bash
uv run python flows/pipeline.py
```

Prefect поднимет сервер, выполнит все таски, покажет логи в консоли и выключится

## Запуск с дашбордом

Терминал 1 — запустить сервер:
```bash
uv run prefect server start
```

Терминал 2 — запустить flow:
```bash
uv run python flows/pipeline.py
```

Дашборд: http://localhost:4200

## CLI (без Prefect)

```bash
uv run python flows/pipeline.py --config configs/flows/fast_test.yaml
```
