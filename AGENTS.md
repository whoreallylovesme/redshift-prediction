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

uv run python -m runtime.pipeline --config examples/configs/teddy.yaml --check
uv run python -m runtime.pipeline --config path/to/experiment.yaml
```

Научные эксперименты запускаются через Prefect командой `uv run python -m runtime.pipeline --config <experiment.yaml>`. Это требование научного руководителя. Подготовка выборок, обучение, прогноз, оценка и отчёт принадлежат одному запуску. Новый эксперимент задаётся YAML с общим блоком `dataset` и словарём `methods`.

Проверка без научных каталогов:

```bash
uv run python -m runtime.pipeline --config examples/configs/euclid.yaml --check
uv run --extra tracking python -m unittest discover -s tests
```

## Архитектура

- `runtime/pipeline.py` — публичный вход Prefect, проверка рецепта и повторный рендер отчёта.
- `lib/config/experiment.py` задаёт схему `preparation`, `dataset` и `methods`; стадии подготовки образуют кешируемый DAG, поля model и params задают метод, единый dataset описывает входы таблиц и изображений.
- `lib/workflow/`, `lib/train/`, `lib/inference/`, `lib/report/` — граф, классический ML, Lightning, расчёт общих метрик и интерактивный HTML.
- `lib/models/nn/` содержит нейросети, `lib/models/tabular/` — классические оценщики. Именованные фабрики доступны через `lib/models/registry.py`.
- `photoz/` содержит предметные правила каталогов, метрик и Phosphoros. Архитектуры с фиксированной схемой входов Euclid находятся в `photoz/models/nn/`.
- `tools/` готовит данные и ресурсы для методов. Форматы научных результатов описаны в `docs/euclid_q1/README.md`.
- Примеры конфигов, правила кеширования и границы трекинга см. в `docs/pipeline.md`. Порядок проверки обучения — в `docs/lightning.md`.
- `ai/skills/` — `run_experiment` для научного запуска, `make_report` для отчёта по эксперименту и `make_monthly_report` для отчёта кафедре. Правила запуска из skills не заменяют схему pipeline.

## Стандарты кода

- Python ≥ 3.13 через `uv`, скрипты запускаются из корня репозитория.
- Строгий mypy; ruff: строка 100, правила `E/F/I/B/UP`.
- `snake_case`; без докстрингов и комментариев; логгер `lib/utils/log.py` вместо `print`.
- Пути только из конфигов, без абсолютных путей в коде.
- Новая модель или опыт не трогает веса, конфиги и прогоны других моделей.

## Подводные камни

- Для нового эксперимента используйте уникальный каталог запуска, созданный Prefect. Снимок параметров сохраняется в `config_snapshot/` этого запуска.
- Методы `kind: tabular` и `kind: lightning` выбирают фабрику через `model`. Параметры модели задаются в `params`, параметры обучения — в `train`.
- Данные в git не коммитятся: `datasets/`, `prepared/`, `runs/`, `external/`, картинки Teddy.
- В конфигах `device: auto` предназначен для synthetic unit-тестов; научный серверный запуск использует явно выбранную CUDA-карту. Прогноз на GPU отличается от CPU в 4-м знаке (TF32/cudnn); побитовые сверки делать на CPU.

## GPU-сервер

**Научные вычисления — только на GPU-сервере.** Обучение, прогноз, оценка научных выборок,
Phosphoros и массовая сборка данных на Mac не запускаются. На Mac разрешены агенты, правка кода и
текстов, git, `--check`, Ruff, mypy, unit-тесты на малых synthetic fixtures, просмотр результатов,
копирование малых артефактов и сборка PDF. Научный запуск выполняется через
`uv run python -m runtime.pipeline --config <experiment.yaml>`; порядок — skill
`ai/skills/run_experiment/`. CI запускает только проверки кода и synthetic-тесты.
Запрошенный пользователем эксперимент не требует повторных approvals на probes, fit, inference,
evaluation, выбор GPU или параллельные сиды внутри прежнего протокола и ресурсного scope. Действовать
самостоятельно; отдельное согласование нужно только для выхода за scope и правил
`docs/gpu_server_manifest.md`.

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
