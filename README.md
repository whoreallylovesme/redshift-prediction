# Регрессия по изображениям и таблицам

В этом репозитории на одном пайплайне сравниваются классические и нейросетевые модели. Основная прикладная задача — оценка фотометрического красного смещения на SDSS Teddy/Happy и Euclid Q1. Обычная регрессия, совместная модель изображений и таблиц, а также сравнение готовых прогнозов используют ту же точку входа.

Из корня репозитория можно запустить полностью локальный пример с Ridge и случайным лесом. Для него все входные данные включены в Git.

```bash
uv sync --frozen
uv run python -m runtime.pipeline --config examples/configs/regression.yaml --check
uv run python -m runtime.pipeline --config examples/configs/regression.yaml
```

Команда печатает путь к интерактивному автономному `report.html`. Для примера со снимками и числовыми признаками замените рецепт на `examples/configs/fusion.yaml`. Для записи параметров, метрик и истории эпох в локальный MLflow выполните

```bash
uv run --extra tracking python -m runtime.pipeline \
  --config examples/configs/fusion.yaml --tracking-uri sqlite:///runs/mlflow.db
```

Рабочий YAML можно хранить вне репозитория. Каждый запуск сохраняет конфигурацию вместе с результатами. Образцы для табличной модели, Lightning и Phosphoros собраны в [examples/configs](examples/configs/).

| Где искать | Содержимое |
| --- | --- |
| [Пайплайн и конфиги](docs/pipeline.md) | Схема эксперимента, граф Prefect, кеширование, MLflow, отчёт и Phosphoros |
| [Ревью репозитория](docs/repository-review.md) | Охват по каталогам, исправленные ошибки, проверки и оставшиеся ограничения |
| [Нейросетевые модели](docs/lightning.md) | Lightning, совместные модели, ResMLP, Transformer и совместимость чекпойнтов |
| [Euclid Q1](docs/euclid_q1/README.md) | Подготовка научных данных, сеток и воспроизведение таблиц публикации |
| [Протокол](docs/experiment_setup.md) | Назначение выборок, валидация и правила сравнений |
| [Инструкции разработки](AGENTS.md) | Команды, проверки и правила работы в репозитории |
