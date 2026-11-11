# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Команды

```bash
# Первичная установка
uv sync
uv run pre-commit install   # хуки ruff + mypy (запускаются через uv)

# Линт и проверка типов (запускать оба перед PR)
uv run ruff check .
uv run mypy .

# Запуск пайплайна через flow-конфиг (оркестрация Prefect)
uv run python runtime/pipeline.py --config configs/flows/fast_test.yaml    # быстрый smoke-тест
uv run python runtime/pipeline.py --config configs/flows/run_all.yaml     # все табличные модели
uv run python runtime/pipeline.py --config configs/flows/train_nn.yaml    # NN пайплайн (Teddy A→B/C/D)
uv run python runtime/pipeline.py --config configs/flows/capsnet_teddy.yaml  # CapsNet 6ch + инференсы B/C/D

# Запуск отдельной стадии (обе CLI-точки принимают только --config)
uv run python runtime/train.py --config configs/nn/capsnet_teddy_w1w2.yaml
uv run python runtime/inference.py --config configs/nn/capsnet_teddy_w1w2.yaml

# Docker
docker build -t photoz:latest .   # CMD внутри — fast_test.yaml flow
```

Тестов пока нет — валидация делается через сами пайплайны. После нетривиальных изменений запускай `fast_test.yaml` как smoke-проверку.

## GPU-сервер (обучение)

```bash
ssh gpu   # <gpu-host>:<port>, kadyrov, ключ <ssh-key>
```

- **4× RTX A6000 48 ГБ**; перед запуском смотри `nvidia-smi` и бери свободную
  (GPU-0 обычно занята кафедральным). В коде задавай `CUDA_VISIBLE_DEVICES=N`.
- **64 ядер, 503 ГБ RAM, /home 44 ТБ** (свободно ~10 ТБ) — датасеты влезают.
- Системный python 3.8 непригоден (проект — py3.13). Окружение через **uv**
  (`~/.local/bin/uv`, managed python 3.13 уже стоит): `uv venv && uv pip install
  torch --index-url https://download.pytorch.org/whl/cu128` — драйвер 12.8,
  дефолтные колёса PyPI (cu13) НЕ работают.
- Docker есть, но демон недоступен без запроса админу.
- Копирование данных: `scp -P <port>` / `rsync -e "ssh -p <port>"`, замер ~8 МБ/с
  с мака (штампы Euclid 1.8 ГБ ≈ 4 мин).
- Правила: сервер общий (студенты + кафедра) — не занимать все GPU, тяжёлые
  раны через tmux/nohup, ничего вне `/home/kadyrov`.
- Гайд сервера (jupyter и т.п.): https://telegra.ph/Gajd-po-rabote-s-serverom-10-26

## Архитектура

Проект — **предсказание фотометрических красных смещений (photoz)**. Оркестратор `runtime/pipeline.py` выбирает между **тремя режимами** по полю `pipeline:` во flow YAML (схема — `lib/config_models.py: FlowConfig`): `tabular`, `nn`, `capsnet`.

### Входные точки

Ровно **две CLI-точки**, обе принимают только `--config` (пути выводятся из конфига или конвенций):
- `runtime/train.py --config <yaml>` — диспетчер по `model_type`: `lightning` (реестр `lib/registry.py` TORCH_TRAINERS: capsnet → свой тренер, остальное → `RegressionModule`) или `sklearn`
- `runtime/inference.py --config <yaml>` — модельно-слепой движок `lib/inference.py` (реестр TORCH_PREDICTORS)

Остальные `runtime/*.py` (`prepare_dataset`, `calc_metrics`, `table`, `plots_html`, `pipeline`) — внутренние таски Prefect, вызываются как функции, без CLI.

### Табличный режим (`pipeline: tabular`)
1. `runtime/prepare_dataset.py` — читает `datasets/dsat.csv`, применяет `configs/data/dataset.yaml`, делает сплит → `prepared/{train,test}.parquet`
2. `runtime/train.py` — грузит табличный YAML-конфиг, запускает `GridSearchCV` → `model.pkl` + JSON-отчёт
3. `runtime/inference.py` — грузит `model.pkl`, предсказывает на test parquet → `preds.parquet`
4. `runtime/calc_metrics.py` — считает метрики → JSON-отчёт
5. `runtime/table.py` / `runtime/plots_html.py` — собирают все ранчи в HTML-сводки

### NN режим (`pipeline: nn`)
Та же форма `train → inference → calc_metrics`, но читает картинки (timm-модели: ConvNeXt, ViT, EfficientNet). Обучение через Lightning `RegressionModule` (`lib/nn/train/regression_lightning.py`).

### CapsNet режим (`pipeline: capsnet`)
Порт Dey+2022 encapZulate-1: `lib/nn/models/capsnet.py` (класс `CapsNet`, 7.9M параметров), Lightning-тренер `lib/nn/train/capsnet_lightning.py` (двухстадийный A/B с непрерывным Adam+ExponentialLR, STOP-файл, resume), labels `lib/nn/train/labels.py` (Sec 4.1 генерация). Лоадеры: `lib/nn/dataloaders/{hdf5,teddy_capsnet,npz,common}.py`. Метрики побитово воспроизводят результаты.

### Конфиг-driven дизайн

Каждая модель полностью описывается YAML-конфигом:
- **Табличные** (`configs/tabular/*.yaml`): `estimator` — Python-строка, `lib/estimator_loader.py` `exec`-ает её для получения sklearn `Pipeline`
- **NN** (`configs/nn/*.yaml`): `model_type: lightning`; capsnet-конфиги дополнительно имеют `model: capsnet`, секции `augment:`, `inference: {splits: [...]}`, `labels:`
- **Flow** (`configs/flows/*.yaml`): конфиг оркестратора — режим, модели, сплиты

Feature-сеты живут в `configs/features/features.yaml` и `configs/features/features_teddy_happy.yaml`.

### Метрики (`lib/metrics.py`)
- `delta = (z_pred - z_true) / (1 + z_true)`
- `bias`: медиана `dz`; `nmad`: нормализованное MAD; `outlier_fraction`: `|dz| > threshold`

Главный OOD-бенчмарк — **Teddy**: тренировка на A, оценка на B/C/D.

## Подводные камни

- **Флоу стирает выходные директории**: `runtime/pipeline.py` в начале делает `shutil.rmtree(runs_root)`. Не направляй `runs_root` на существующие артефакты.
- **`estimator` в YAML исполняется** через `exec` — это код.
- **Gitignore-объёмы**: `datasets/`, `prepared/`, `runs*/`, `*_grizw1w2/`, `nn_*/` — данные, не коммитить.
- **Железо**: MacBook M3 → PyTorch `mps` backend; в конфигах `device: auto`.
- `python_sh/` — разовые экспериментальные скрипты без контракта; `tools/download_imgs/` — скачивание картинок; `tools/capsnet/` — утилиты капснета.

## Стандарты кода

- Строгий mypy; ruff: строка 100, `E/F/I/B/UP`, py313.
- Python ≥ 3.13 через `uv`. Скрипты из корня репо.
- `snake_case`; коммиты — короткие imperative (< 72). Без докстрингов и комментариев. Логгер вместо print (`lib/log.py`, префект-совместимый).
