# Photo-z: прогноз красного смещения по фотометрии

Курсовая работа и ВКР (ВМК МГУ, кафедра ИИТ). Модели photo-z на SDSS Teddy/Happy и Euclid Q1.

```bash
uv sync
uv run python runtime/pipeline.py --config configs/flows/fast_test.yaml
```

| Что | Где |
|---|---|
| Команды, архитектура, правила разработки | `AGENTS.md` |
| Euclid Q1: итоговая модель и воспроизведение чисел отчётов | `docs/euclid_q1/README.md` |
| Протокол экспериментов | `docs/experiment_setup.md` |
| Правила GPU-сервера | `docs/gpu_server_manifest.md` |
| Отчёты научным руководителям и кафедре | `reports/weekly/`, `reports/monthly/` |
| Инструкции и скиллы агентов | `ai/` |
