# Тексты работ

## Курсовая, май 2026

«Прогноз космологического красного смещения по фотометрическим признакам».

- [Курсовая работа](coursework/coursework.pdf) — архивный PDF, 40 страниц.
- [Презентация](coursework/presentation.pdf) — архивный PDF, 39 слайдов.
- [Исходник LaTeX](coursework/coursework.tex), [58 рисунков](coursework/figures/) и [эмблема МГУ](coursework/msu.jpg).

## Сборка курсовой

Потребуются Tectonic и системные шрифты Times New Roman и Courier New.
Выполните из корня репозитория:

```sh
cd manuscripts/coursework
mkdir -p build
tectonic -X compile --outdir build coursework.tex
```

Собранный PDF появится в `manuscripts/coursework/build/coursework.pdf`. Архивный PDF
хранится отдельно в `manuscripts/coursework/coursework.pdf`.
