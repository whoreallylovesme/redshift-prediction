# AGENTS.md

Единые инструкции для кодовых агентов в этом репозитории. Codex и Kimi читают этот файл напрямую,
Claude Code — через `CLAUDE.md`, который его импортирует. Правила меняются только здесь.

## Проект

Предсказание фотометрических красных смещений (photo-z). Две линии работ:

- **SDSS Teddy/Happy** — табличные модели, CNN/ViT (timm), CapsNet (порт Dey+2022). Главный
  OOD-бенчмарк: обучение на Teddy A, оценка на B/C/D.
- **Euclid Q1** — CLIP-предобучение на 1,18 млн объектов полей F/S, дообучение (итоговая модель
  P_final), сравнение с Phosphoros и PHZ Euclid. Поля F/S — обучение, поле N — только тест.
  Порядок запуска и происхождение каждого числа отчётов — `docs/euclid_q1/README.md`,
  постановка опытов — `docs/experiment_setup.md`.

## Команды

```bash
uv sync
uv run pre-commit install          # хуки ruff + mypy
uv run ruff check .                # перед каждым коммитом
uv run mypy .

uv run python runtime/pipeline.py --config configs/flows/fast_test.yaml   # smoke-тест
uv run python runtime/pipeline.py --config configs/flows/<flow>.yaml
```

**Всё запускается только через Prefect** — требование научного руководителя. Обучение, прогноз,
оценка, отчёты и любые опыты запускаются флоу Prefect (`runtime/pipeline.py --config <flow>.yaml`).
Новый опыт — это новый flow-YAML или новая стадия во флоу, а не отдельный скрипт, запущенный
руками. Прямые вызовы `runtime/train.py` и `runtime/inference.py` (Hydra:
`model=<...> dataset=<...>`) — только для отладки одной стадии; результаты таких запусков в отчёты
не идут.

Проверка без научных каталогов:

```bash
uv run python -m runtime.pipeline --config configs/flows/regression_example.yaml --check
uv run python -m runtime.pipeline --config configs/flows/regression_example.yaml
uv run --extra tracking python -m unittest discover -s tests
```

## Архитектура

- `runtime/pipeline.py` — публичный вход Prefect, проверка рецепта и повторный рендер отчёта.
- `lib/config/experiment.py` задаёт схему `dataset` и `methods`; тип метода определяет необходимые поля.
- `lib/workflow/`, `lib/train/`, `lib/inference/`, `lib/report/` — граф, классический ML, Lightning, расчёт общих метрик и интерактивный HTML.
- `photoz/` содержит предметные правила каталогов, моделей, метрик и Phosphoros. Исторические формы `pipeline:` обрабатывает `photoz/legacy_flow.py`.
- `tools/` готовит данные и ресурсы для методов. Форматы научных результатов описаны в `docs/euclid_q1/README.md`.
- Примеры конфигов, правила кеширования и границы трекинга см. в `docs/pipeline.md`. Порядок проверки обучения — в `docs/lightning.md`.

## Стандарты кода

- Python ≥ 3.13 через `uv`, скрипты запускаются из корня репозитория.
- Строгий mypy; ruff: строка 100, правила `E/F/I/B/UP`.
- `snake_case`; без докстрингов и комментариев; логгер `lib/utils/log.py` вместо `print`.
- Пути только из конфигов, без абсолютных путей в коде.
- Новая модель или опыт не трогает веса, конфиги и прогоны других моделей.

## Подводные камни

- Для нового эксперимента используйте уникальный каталог запуска, созданный Prefect. Старые научные рецепты имеют собственные правила возобновления.
- Поставляемые YAML выбирают именованные фабрики оценщиков. Сохранённые пользовательские рецепты с Python в поле `estimator` допускаются только из доверенного источника.
- Данные в git не коммитятся: `datasets/`, `prepared/`, `runs/`, `external/`, картинки Teddy.
- Mac M3 — бэкенд `mps`, в конфигах `device: auto`. Прогноз на GPU отличается от CPU в 4-м знаке
  (TF32/cudnn); побитовые сверки делать на CPU.

## GPU-сервер

`ssh gpu`: 4× RTX A6000, сервер общий. Перед запуском — свободная карта по `nvidia-smi`
(GPU 0 обычно занята), `CUDA_VISIBLE_DEVICES=N`, долгие прогоны в `tmux` с `nice`, без `pkill -f`.
Все правила — `docs/gpu_server_manifest.md`.

## Git и публикация

- Коммит или push — только по явной просьбе пользователя. Хуки pre-commit не обходить.
- Сообщения коммитов — короткие, в повелительном наклонении, до 72 символов.
- Репозиторий публичный: никаких адресов серверов, портов, ключей, токенов и персональных данных
  научных руководителей и соавторов.

## Отчёты и тексты

- LaTeX собирается XeLaTeX/tectonic (`fontspec` + `polyglossia`); `pdflatex` не подходит.
- Кавычки только «ёлочки», `<<` и `>>` рендерятся буквально; тире — `—`.
- Числа в отчётах берутся из артефактов (`facts.json`, `metrics/*.json`), а не переписываются руками.
- В отчётах и планах ИИ-агенты не упоминаются, кроме задач, где это предмет работы.
