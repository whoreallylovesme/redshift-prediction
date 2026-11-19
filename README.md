# Регрессия по изображениям и таблицам

В этом репозитории на одном пайплайне сравниваются классические и нейросетевые модели. Основная прикладная задача — оценка фотометрического красного смещения на SDSS Teddy/Happy и Euclid Q1. Обычная регрессия, совместная модель изображений и таблиц, а также сравнение готовых прогнозов используют ту же точку входа.

Один YAML описывает датасет и все методы эксперимента. В [примере TEDDY](examples/configs/teddy.yaml) CatBoost и CapsNet обучаются по каталогу A и оцениваются на одних и тех же объектах B. В [примере Euclid](examples/configs/euclid.yaml) Phosphoros получает подготовленную фотометрию. Для этих запусков нужны локальные научные данные.

Для знакомства с пайплайном выполните из корня репозитория:

```bash
uv sync --frozen
uv run python -m runtime.pipeline --config examples/configs/regression.yaml --check
uv run python -m runtime.pipeline --config examples/configs/regression.yaml
```

Пример обучает Ridge и CatBoost на 442 реальных наблюдениях из комплекта scikit-learn. После установки зависимостей он работает офлайн и печатает путь к интерактивному HTML. Описание целевой величины и признаков приведено в [паспорте датасета](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).

Чтобы сохранить параметры и метрики в локальный MLflow, добавьте `--tracking-uri`:

```bash
uv run --extra tracking python -m runtime.pipeline \
  --config examples/configs/regression.yaml --tracking-uri sqlite:///runs/mlflow.db
```

Рабочий YAML можно хранить вне Git. Каждый запуск сохраняет его разрешённую конфигурацию и параметры моделей вместе с результатами.

| Где искать | Содержимое |
| --- | --- |
| [Пайплайн и конфиги](docs/pipeline.md) | Схема эксперимента, граф Prefect, кеширование, MLflow, отчёт и Phosphoros |
| [Ревью репозитория](docs/repository-review.md) | Охват по каталогам, исправленные ошибки, проверки и оставшиеся ограничения |
| [Нейросетевые модели](docs/lightning.md) | Lightning, совместные модели, ResMLP, Transformer и совместимость чекпойнтов |
| [Euclid Q1](docs/euclid_q1/README.md) | Подготовка научных данных, сеток и воспроизведение таблиц публикации |
| [Протокол](docs/experiment_setup.md) | Назначение выборок, валидация и правила сравнений |
| [Инструкции разработки](AGENTS.md) | Команды, проверки и правила работы в репозитории |
