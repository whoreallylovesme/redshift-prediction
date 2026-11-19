# Регрессия по изображениям и таблицам

В этом репозитории на одном пайплайне сравниваются классические и нейросетевые модели. Основная прикладная задача — оценка фотометрического красного смещения на SDSS Teddy/Happy и Euclid Q1. Обычная регрессия, совместная модель изображений и таблиц, а также сравнение готовых прогнозов используют ту же точку входа.

Из корня репозитория можно запустить полностью локальный пример с Ridge и случайным лесом. Для него все входные данные включены в Git.

```bash
uv sync --frozen
uv run python -m runtime.pipeline --config configs/flows/regression_example.yaml --check
uv run python -m runtime.pipeline --config configs/flows/regression_example.yaml
```

Команда печатает путь к интерактивному автономному `report.html`. Для примера со снимками и числовыми признаками замените рецепт на `configs/flows/fusion_example.yaml`. Для записи параметров, метрик и истории эпох в локальный MLflow выполните

```bash
uv run --extra tracking python -m runtime.pipeline \
  --config configs/flows/fusion_example.yaml --tracking-uri sqlite:///runs/mlflow.db
```

| Где искать | Содержимое |
| --- | --- |
| [Пайплайн и конфиги](docs/pipeline.md) | Схема эксперимента, граф Prefect, кеширование, MLflow, отчёт и Phosphoros |
| [Ревью репозитория](docs/repository-review.md) | Охват по каталогам, исправленные ошибки, проверки и оставшиеся ограничения |
| [Нейросетевые модели](docs/lightning.md) | Lightning, совместные модели, ResMLP, Transformer и совместимость чекпойнтов |
| [Euclid Q1](docs/euclid_q1/README.md) | Подготовка научных данных, сеток и воспроизведение таблиц публикации |
| [Протокол](docs/experiment_setup.md) | Назначение выборок, валидация и правила сравнений |
| [Инструкции разработки](AGENTS.md) | Команды, проверки и правила работы в репозитории |
