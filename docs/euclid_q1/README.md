# Euclid Q1: фотометрические красные смещения, финальная модель P_final

Здесь описано, как из публичных данных Euclid Q1 получить финальную модель P_final и числа отчёта
`reports/weekly/2026-11-06` (и раздела 2.1–2.5 месячного отчёта
`reports/monthly/2026-11-12`).

Цепочка такая:

1. Подготовка данных: пакеты HDF5 с картинками 64×64 (VIS/Y/J/H) и 14-полосной таблицей, метки,
   выборка из 1 млн объектов F/S для предобучения, тестовая выборка HETDEX.
2. Phosphoros 2.2.1: эталонный шаблонный метод для сравнения и источник «безметочного» z, по
   которому фотометрия F/S переводится в полосы UNIONS.
3. Перевод фотометрии F/S в систему UNIONS: шаблонные сетки, нуль-пункты Euclid (подогнаны только
   на F), таблицы `translated_*.npz`.
4. Модель: CLIP-предобучение на 1 181 164 объектах F/S (2 сида), дообучение на 21 156 метках F/S
   (6 пар сидов), предсказания для N, D2–D5, HETDEX, классические базовые модели на тех же
   21 156 строках.
5. Отчёт: `facts.json`, таблицы `tables/*.tex`, рисунки `figures/*.pdf` и файлы
   `metrics/results_*.json` строятся из предсказаний этим же кодом.

Метки поля N (EDF-N) в обучении не используются: на них только считаются метрики.

## Что нужно установить

- Python 3.13 и `uv`: `uv sync` в корне репозитория. Все скрипты запускаются из корня.
- GPU с CUDA для предобучения и дообучения (исходный прогон: RTX A6000, TF32 включён).
- **Phosphoros 2.2.1** (внешняя программа на C++, не pip-пакет). Использовалась сборка conda
  `phosphoroscore=2.2.1` из канала `astrorama` (py311, osx-arm64; alexandria 2.33.0,
  elements 6.3.7). Окружение ставится в `external/phosphoros/env`:

  ```bash
  conda create -p external/phosphoros/env -c astrorama -c conda-forge phosphoroscore=2.2.1
  ```

  Нужны также архив `AuxiliaryData_0.8.tar.xz` (кладётся в `external/phosphoros/`) и рабочая
  директория севера `datasets/euclid_q1/phosphoros/workspace_north/AuxiliaryData`. В ней должны
  быть фильтры UNIONS и Euclid, 31 SED `COSMOS_public_el_sigma200` (COSMOS с эмиссионными линиями,
  получены официальным скриптом AEL), кривые поглощения (`SB_Calzetti`, `SB_Calzetti_mod`,
  `SMC_Prevot`, `Gordon23/G23`), солнечный спектр и функция светимости
  `PublicQ1/Mr_flat_minus24_to0`. Конфиги Phosphoros лежат в `configs/euclid_q1/phosphoros/*.conf`.
  Путей в них нет: `phosphoros-root`, входной каталог и выходная директория передаются из
  командной строки, а sha256 файла `compute_redshifts_*.conf` записывается в результат как
  `config_id`.
- `aws` CLI или любой HTTP-клиент для скачивания каталогов Q1 из публичного бакета
  `s3://nasa-irsa-euclid-q1`.

Место на диске: мозаики MER для выборки 1M около 1,6 ТБ (читаются один раз), шарды 64×64 около
94 ГБ, `stage` около 19,5 ГБ, мозаики HETDEX около 0,9 ТБ (после нарезки их можно удалить).

## Раскладка данных

Все пути в конфигах относительные и начинаются с `datasets/euclid_q1/` (данные) или
`runs/euclid_clip_p_final/` (результаты модели). Обе директории и `external/` в `.gitignore`.

| Путь | Что это | Откуда |
|---|---|---|
| `mer/EUC_MER_FINAL-CAT_TILE*.fits` | каталоги MER Q1 (EDF-N/F/S) | шаг 1 |
| `phz/EUC_PHZ_PHZCAT__*.fits` | официальные PHZ Q1 | шаг 1 |
| `edf_f_tiles.txt`, `edfn_tiles.txt` | id тайлов EDF-F и EDF-N, по одному в строке | список тайлов Q1 |
| `registry/manifest.json`, `registry/registry/*.parquet` | реестр объектов F/S по тайлам MER | внешний вход |
| `pretrain/guards.npz` | запрещённые тайлы, id и позиции для отбора 1M | внешний вход |
| `products/<набор>/` | вырезки MER по объектам (`*.npz` + `*.json`), реестры и манифесты | внешний вход |
| `inputs/raw14.npz`, `inputs/raw14_expansion.npz` | 14-полосная фотометрия строк пилотного и расширенного пакетов | внешний вход |
| `inputs/table_descriptor.npy` | дескрипторы полос 14×18 | внешний вход |
| `labels/f_fitting.npz` | метки F (TRAIN/VALID) | внешний вход |
| `labels/threedhst/` | 3D-HST v4.1.5 GOODS-S (`goodss_3dhst.v4.1.5.IR.cat`, `goodss_3dhst.v4.1.5.zfit.linematched.fits`) | публичный каталог |
| `labels/labels_release.parquet`, `labels/desi_redshifts.parquet`, `labels/n_native_photometry.npz`, `labels/nb_native_photometry.npz` | метки и фотометрия поля N (N-A, N-B) | внешний вход |
| `labels/reference_phz.parquet`, `labels/ho_native_photometry.npz` | метки HO (MUSE, SDSS) и их фотометрия | внешний вход |
| `labels/{t1,d2,d3,d4,d5,da}/`, `labels/exclusions/`, `labels/edfn_catalog_accuracy.parquet` | наборы меток D2–D5 (`<набор>_ids.npy`, `<набор>_labels.parquet`) и наборы для исключения из HETDEX | внешний вход |
| `hetdex/hetdex_sc2_v1.5.fits` | каталог HETDEX HPSC-2 (Zenodo 10.5281/zenodo.19581262) | публичный каталог |
| `packets/*.h5`, `packets/normalizers.json` | пакеты модели и замороженные нормировки | шаг 3 |

«Внешний вход» — артефакт, который этот MR не строит: он получен раньше (скачивание вырезок,
сбор меток DESI/SDSS/MUSE, реестр объектов). Его кладут по указанному пути как есть.

## Порядок запуска

Ниже `T=tools/euclid_q1` и `C=configs/euclid_q1`.

```bash
T=tools/euclid_q1
C=configs/euclid_q1
uv sync
```

### 1. Каталоги Q1

```bash
aws s3 cp --no-sign-request --recursive s3://nasa-irsa-euclid-q1/q1/catalogs/MER_FINAL_CATALOG/ \
  datasets/euclid_q1/mer_tiles/ --exclude "*" --include "*EUC_MER_FINAL-CAT_TILE*"
mkdir -p datasets/euclid_q1/mer
find datasets/euclid_q1/mer_tiles -name "*.fits" -exec mv {} datasets/euclid_q1/mer/ \;
aws s3 cp --no-sign-request --recursive s3://nasa-irsa-euclid-q1/q1/catalogs/PHZ_PF_OUTPUT_CATALOG/ \
  datasets/euclid_q1/phz_tiles/ --exclude "*" --include "*EUC_PHZ_PHZCAT__*"
mkdir -p datasets/euclid_q1/phz
find datasets/euclid_q1/phz_tiles -name "*.fits" -exec mv {} datasets/euclid_q1/phz/ \;
```

### 2. Пакет с метками F и нормировки

```bash
uv run python $T/build_packets.py labelled --config $C/build_packets_labelled.yaml
```

Пишет `packets/dataset.h5` с ролями F TRAIN 17 809, F VALID 1 101, N-A 7 253 и SSL 30 000.
Рядом пишется `packets/normalizers.json`: нормировки картинок, таблицы и спектров,
подогнанные только на F TRAIN и SSL. Все следующие пакеты используют эти нормировки без
переподгонки (в исходном прогоне sha256 c951e169…).

### 3. Метки CDFS и остальные пакеты

```bash
uv run python $T/build_labels.py cdfs --config $C/build_labels_cdfs.yaml
for s in cdfs ho nb d2 d3 d4 d5; do
  uv run python $T/build_packets.py stamps --config $C/build_packets_stamps_$s.yaml
done
uv run python $T/build_packets.py ssl --config $C/build_packets_ssl_extra.yaml
uv run python $T/build_packets.py ssl --config $C/build_packets_ssl_extra2.yaml
```

`build_labels cdfs` сопоставляет 3D-HST (спектральные и точные грисмовые z) с MER EDF-F: совпадение
не дальше 0,5″, следующий сосед не ближе 1,0″. Из выборки исключаются объекты пакета F и HO.
Результат: `labels/cdfs/{cdfs_train.parquet, cdfs_native_photometry.npz}`.

Пакеты из вырезок:
- `cdfs.h5` (2 130) и `ho.h5` (1 253) участвуют в обучении;
- `nb.h5` (N-B, 29 623) и `d2…d5.h5` нужны только для предсказаний;
- `extra.h5` и `extra2.h5` дают старые строки SSL для предобучения.

### 4. PHZ Euclid и таблицы меток

```bash
uv run python $T/phz_lookup.py --config $C/phz_lookup.yaml
uv run python $T/build_labels.py north --config $C/build_labels_north.yaml
uv run python $T/build_labels.py translation --config $C/build_labels_translation.yaml
```

`phz_lookup` выбирает `PHZ_MEDIAN`, `PHZ_MODE_1`, `PHZ_70_INT` и `PHZ_CLASSIFICATION` из
официальных каталогов PHZ. Id берутся из пакетов шагов 2–3: `phz_lookup.npz` строится по строкам
F TRAIN, F VALID и N-A пакета `dataset.h5` и всем строкам `nb.h5`, `phz_{ho,cdfs,d2,d3,d4,d5}.npz`
— по всем строкам соответствующих пакетов. Источники id задаются в конфиге списком `ids`: файлы
`.h5` (с необязательным `indices`), `.npy`, `.npz` или `.parquet`. В исходном прогоне список id
для `phz_lookup.npz` содержал ещё 1 658 объектов вне этих строк; их значения ни один из
следующих шагов не читает.

`north` пишет `labels/labelled_n_table.parquet`: 36 876 надёжных меток N (N-A 7 253 и
N-B 29 623) с тайлом, VIS и PHZ Euclid. `translation` пишет `translation/inputs/f_labels.npz`:
метки F TRAIN, F VALID, CDFS, MUSE и HO-F SDSS для перевода фотометрии.

### 5. Выборка 1M для предобучения

```bash
uv run python $T/select_pretrain_objects.py select --config $C/select_pretrain_objects.yaml
uv run python $T/select_pretrain_objects.py photometry --config $C/select_pretrain_objects.yaml
uv run python $T/download_mosaics.py discover --config $C/download_mosaics.yaml
uv run python $T/download_mosaics.py fetch --config $C/download_mosaics.yaml
uv run python $T/cut_stamps.py prepare --config $C/cut_stamps.yaml
uv run python $T/cut_stamps.py cut --config $C/cut_stamps.yaml
uv run python $T/build_pretrain_store.py index --config $C/build_pretrain_store.yaml
uv run python $T/build_pretrain_store.py stage-old --config $C/build_pretrain_store.yaml
```

Что делают эти команды:

- `select` отбирает 500 000 объектов F и 500 000 объектов S без меток. Отбор детерминированный:
  splitmix64 от `object_id` с сидом 20261101. Исключаются запрещённые тайлы и id, объекты
  ближе 10″ к защищённым позициям и объекты, у которых есть сосед ближе 1″.
- `photometry` пишет 14-полосную фотометрию MER в `pretrain/raw14/`.
- `discover` и `fetch` скачивают мозаики VIS/Y/J/H (science, rms, flag) по тайлам-владельцам с
  зеркала IRSA на S3.
- `cut` нарезает штампы 64×64 и сразу нормирует их замороженными нормировками. Результат: 104
  шарда float16. `cut` можно запускать по частям: `--key <ключ>` (повторяемый), готовые шарды
  пропускаются. Флаговые плоскости не читаются (`read_flags: false`). Это эквивалентно исходному
  прогону: бит INVALID в Q1 не выставлен.
- `index` индексирует шарды и считает `ssl_keep`. Из предобучения исключаются тайлы F VALID,
  отложенные тайлы пилота, объекты HO и объекты, которые уже есть в старых источниках.
- `stage-old` выгружает 181 164 старые строки SSL в `pretrain/stage`: 27 219 из пилота, 53 449 из
  `extra.h5` и 100 496 из `extra2.h5`.

В итоге для предобучения доступно 1 181 164 строки.

### 6. Тестовая выборка HETDEX (H6)

```bash
uv run python $T/hetdex.py select --config $C/hetdex.yaml
uv run python $T/hetdex.py native --config $C/hetdex.yaml
uv run python $T/hetdex.py mosaics --config $C/hetdex.yaml
uv run python $T/hetdex.py cut --config $C/hetdex.yaml
uv run python $T/hetdex.py packet --config $C/hetdex.yaml
uv run python $T/hetdex.py phosphoros-request --config $C/hetdex.yaml
uv run python $T/phz_lookup.py --config $C/phz_lookup_h6.yaml
```

`select` отбирает источники HETDEX HPSC-2 по таким правилам:

- поле NEP, типы `oii`/`lae`/`agn`/`lzg`, `p_conf ≥ 0,5` и `p_cnn ≥ 0,5`;
- однозначное совпадение с MER EDF-N: не дальше 1,0″, следующий сосед не ближе 1,5″;
- исключаются все использованные наборы меток, по id и в радиусе 1″, в том числе таблица меток N
  из шага 4.

Красные смещения записываются только в `build/h6_labels.parquet`. `packet` пишет
`packets/h6.h5` (18 155 строк). 16 id из `exclude_ids` в пакет не входят: для них в исходном
прогоне не было вырезок. `mosaics` и `cut` принимают `--tile <id>` для работы по частям.
Последняя команда пишет PHZ Euclid для id из `build/h6_ids.npy` в `phz_lookups/phz_h6.npz`.

### 7. Фотометрия MER

```bash
uv run python $T/extract_mer_photometry.py --config $C/extract_mer_photometry.yaml
```

`extract_mer_photometry` пишет `native/<SET>_native_photometry.npz`. Это 14 полос в апертуре
2FWHM плюс `GAL_EBV`, детекционный и NIR-поток, для наборов F_TRAIN, F_VALID, N_A (тестовые
строки N пакета `dataset.h5`), CDFS, HO, D2–D5 и H6.

### 8. Phosphoros

```bash
uv run python $T/phosphoros.py workspace --config $C/phosphoros.yaml
uv run python $T/phosphoros.py grids --side north --config $C/phosphoros.yaml
uv run python $T/phosphoros.py grids --side south --config $C/phosphoros.yaml
uv run python $T/phosphoros.py prepare --config $C/phosphoros.yaml
uv run python $T/phosphoros.py run --config $C/phosphoros.yaml
uv run python $T/phosphoros.py postprocess --config $C/phosphoros.yaml
```

Что делают эти команды:

- `workspace` собирает рабочую директорию юга (DECam g,r,i,z + Euclid) из рабочей директории
  севера и архива AuxData 0.8.
- `grids` строит сетки моделей (31 SED, z от 0 до 6 с шагом 0,01, 767 моделей на каждое z), сетку
  галактического поглощения Gordon23 и выгрузку сетки в FITS.
- `prepare` переводит апертурные потоки в полные и умножает их на нуль-пункты Tucci+2026
  (таблица 1). Ошибки получают пол `hypot(σ, β·F)`. В движок идут объекты с двумя и более
  валидными полосами и конечным `GAL_EBV`.
- `run` вызывает `PhosphorosComputeRedshifts` (BAYESIAN, априорные объём 0,3 и плоская функция
  светимости).
- `postprocess` берёт медиану апостериорного Z-1D-PDF: первый узел, где CDF ≥ 0,5. Результат:
  `phosphoros/out/predictions/<SET>.parquet`, колонка `ppub_median`.

Север: N_A, N_B, D2–D5, H6. Юг: F_TRAIN, F_VALID, HO, CDFS.

### 9. Перевод фотометрии F/S в полосы UNIONS

```bash
uv run python $T/build_template_grids.py --config $C/build_template_grids.yaml
uv run python $T/fit_zero_points.py --config $C/fit_zero_points.yaml
uv run python $T/phosphoros.py run --variant south_zpf --config $C/phosphoros.yaml
uv run python $T/phosphoros.py postprocess --variant south_zpf --config $C/phosphoros.yaml
uv run python $T/translate_photometry.py --config $C/translate_photometry.yaml
```

Шаги:

1. `build_template_grids` выгружает сетки севера и юга (767 моделей × 601 z) в
   `translation/grids/`.
2. `fit_zero_points` подгоняет множители нуль-пунктов VIS/Y/J/H только по меткам F TRAIN: подгонка
   шаблонов с исключением одной полосы, две итерации, бутстрап по тайлам, вложенный бутстрап на
   200 повторов. Результат: `translation/zero_points/zpc_f.json`, ключ `factors.euclid`. В
   исходном прогоне VIS 1,0480, Y 0,9623, J 0,9658, H 0,9312.
3. Вариант Phosphoros `south_zpf` пересчитывает z юга с этими множителями вместо Tucci. Метки при
   этом не используются: это и есть безметочное `z_fix`.
4. `translate_photometry` при `z_fix` подгоняет лучший шаблон юга, предсказывает потоки UNIONS
   u,g,r,i,z, разыгрывает шум u и собирает 14-слотовую таблицу с замороженными нормировками.
   Результат: `translation/tables/translated_{F_TRAIN,F_VALID,CDFS,MUSE}.npz`. Именно эти таблицы
   подставляются вместо исходных при дообучении.

Для сравнения «Phosphoros без информации о N» в отчёте нужен ещё вариант севера:

```bash
uv run python $T/phosphoros.py run --variant north_zpf --config $C/phosphoros.yaml
uv run python $T/phosphoros.py postprocess --variant north_zpf --config $C/phosphoros.yaml
```

### 10. Модель: предобучение, дообучение, предсказания, классические модели, отчёт

```bash
uv run python runtime/pipeline.py --config configs/flows/euclid_clip_p_final.yaml
```

Флоу `pipeline: euclid_clip` выполняет пять этапов по порядку:

1. **Предобучение CLIP** (`configs/nn/euclid_clip_pretrain.yaml`, сиды 20261025 и 20261026,
   12 эпох, батч 256, AdamW 3e-4/1e-4, 5 эпох прогрева, затем косинус, клиппинг 5, D4-аугментации,
   InfoNCE с T = 0,1). Результат: `runs/euclid_clip_p_final/clip_1m_<сид>/clip_state.pt` и
   `ssl_history.json`. Около 3 ч на сид на A6000.
2. **Дообучение P_final** (`configs/nn/euclid_clip_p_final.yaml`) для пар (сид CLIP, сид
   дообучения):
   (20261025, 20261025), (20261026, 20261026), (20261025, 20261027), (20261026, 20261028),
   (20261025, 20261029), (20261026, 20261030).

   Рецепт:
   - голова на 200 бинов log(1+z) с мягкой целью;
   - веса строк (плотность log(1+z))^−0,5 с ограничением 10;
   - все 14 полос, 40 эпох, батч 128;
   - обучающие метки: F TRAIN, CDFS 2 130, MUSE 800, HO-S SDSS 387;
   - z-равномерная валидация на 1 071 объекте (сид 20261106), остальные 21 156 строк идут в
     обучение;
   - таблицы F/S заменяются переведёнными из шага 9.

   Результат в `runs/euclid_clip_p_final/P_final/`: `P_final_<сид>_state.pt`,
   `P_final_<сид>_pred.npz` и `P_final_<сид>.json` (рецепт, история обучения, id и z валидации,
   метрики). Около 18 мин на сид.
3. **Предсказания** из сохранённых весов: `runs/euclid_clip_p_final/inference/P_final_<сид>_pred.npz`.
4. **Классические модели** (`configs/nn/euclid_classical.yaml`): CatBoost, Random Forest, ANNz,
   LLR (k-NN, k = 1000) и GAM. Признаки — 9 переведённых полос (значения, маски), строки обучения
   те же 21 156, валидация та же. Результат:
   `runs/euclid_clip_p_final/classical/<модель>/{predictions.npz, complete.json}`.
5. **Отчёт** (`configs/euclid_q1/report.yaml`), см. шаг 11. Результат:
   `runs/euclid_clip_p_final/report/`.

Флоу ничего не удаляет. Этап пропускается, если его результат уже есть: `clip_state.pt`,
`P_final_<сид>.json`, `inference/P_final_<сид>_pred.npz`, `classical/<модель>/complete.json` или
`report/facts.json`.
Чтобы пересчитать этап, его результат удаляют вручную. Каждый этап можно отключить в
`configs/flows/euclid_clip_p_final.yaml`:

- `pretrain.enabled: false` вместе с `finetune.clip_root: <директория с clip_1m_<сид>/>` — взять
  готовые веса CLIP;
- `finetune.enabled: false` вместе с `finetune.out_dir: <директория с P_final_<сид>_state.pt>` —
  только предсказания из готовых весов;
- `inference: false`;
- `classical.enabled: false`;
- `report.enabled: false`.

Конфиг каждого запуска с подставленными сидом и путями сохраняется в
`runs/euclid_clip_p_final/configs/`. Любой этап можно повторить отдельно:

```bash
uv run python -c "from runtime.train import train; train('runs/euclid_clip_p_final/configs/P_final_20261027.yaml')"
uv run python -c "from lib.inference import inference; inference(config_path='runs/euclid_clip_p_final/configs/P_final_inference.yaml')"
```

В YAML моделей `model_type: torch` и `model: <ключ>` выбирают тренер из реестра
(`euclid_clip_pretrain`, `euclid_clip`, `euclid_classical`) и предиктор `euclid_clip`.

### 11. Отчёт

Отчёт строится последним этапом того же флоу:

```bash
uv run python runtime/pipeline.py --config configs/flows/euclid_clip_p_final.yaml
```

Этап вызывает `runtime/euclid_report.build_report` с конфигом
`runs/euclid_clip_p_final/configs/report.yaml`. Это `configs/euclid_q1/report.yaml`, в котором
флоу подставляет тег, сиды, директории метаданных (`P_final/`), предсказаний (`inference/`, а при
`inference: false` — `P_final/`), весов CLIP (`clip_1m_<сид>/ssl_history.json`), классических
моделей и `out_dir`. Повторить только отчёт (файлы в `report/` перезаписываются):

```bash
uv run python -c "from pathlib import Path; from runtime.euclid_report import build_report; build_report(Path('runs/euclid_clip_p_final/configs/report.yaml'))"
```

На одном ядре CPU отчёт строится около 16 мин, почти всё время уходит на бутстрап.

Входы (все пути в `configs/euclid_q1/report.yaml`):

- `P_final_<сид>.json` (история обучения, id и z валидации 1 071) и `P_final_<сид>_pred.npz`;
- `classical/<модель>/predictions.npz`;
- Phosphoros: `phosphoros.files` — предсказания `phosphoros/out/predictions/<SET>.parquet` для
  CDFS, D2–D5, F_TRAIN, F_VALID, HO, N_A, N_B, `phosphoros.hetdex` — H6. Phosphoros без N
  (`phosphoros_zpf`): север из варианта `north_zpf`, юг — те же файлы, что у Phosphoros. Файлы
  читаются в порядке имён, при повторе `object_id` берётся первое значение;
- PHZ Euclid: `phz_lookups/phz_{lookup,cdfs,ho,d2,d3,d4,d5,h6}.npz`, для N — колонка `phz`
  таблицы `labelled_n_table.parquet`;
- метки: `labelled_n_table.parquet`, `labels/<набор>/<набор>_labels.parquet` (колонка `z_desi`
  для D2–D4, `z_ext` для D5), `hetdex/build/h6_{labels,meta}.parquet`, `cdfs_train.parquet`,
  `reference_phz.parquet` и `ho_native_photometry.npz`;
- тайлы — `owner_tile` пакетов; счётчики выборки 1M — `pretrain/store/rows.npz`.

Результаты в `runs/euclid_clip_p_final/report/`:

| Файл | Что внутри | Код |
|---|---|---|
| `facts.json` | все метрики, интервалы, бины, кривые обучения, счётчики меток | `runtime/euclid_report.py` |
| `tables/metrics.tex`, `tables/metrics_delta.tex` | таблица метрик и ΔNMAD с 95 % ДИ | `metrics_tables` |
| `tables/classical.tex` | классические модели против нашей | `classical_table` |
| `figures/fig_{dist,scatter_N,scatter_Nall,modes,bins,baselines,train}.pdf` | рисунки отчёта | `lib/visual/euclid_report.py` |
| `metrics/results_final_vs_{pub,zpf}.json` | описательная оценка по наборам с бутстрапом по тайлам и сидам | `lib/metrics/euclid_eval.comparison_results` |
| `metrics/results_h6_P_final.json` | HETDEX [OII]: модель и гибрид с Phosphoros | `lib/metrics/euclid_eval.hetdex_results` |

## Форматы результатов

- `P_final_<сид>_pred.npz`: для каждого набора `<set>` ∈ {`valid`, `north`, `nb`, `ho`, `d2`,
  `d3`, `d4`, `d5`, `h6`, `zval`} есть ключи `<set>_object_id`, `<set>_base`, `<set>_mu`, `<set>_s`
  и `<set>_phz_ok`. Здесь `north` — N-A, `valid` — F VALID, `zval` — валидация 1 071. Оценка
  z = expm1(`base` + `mu`), где `mu` — медиана распределения по бинам log(1+z), `s` — логарифм
  полуширины 16–84 %. Предсказание ансамбля — среднее z по 6 сидам.
- `classical/<модель>/predictions.npz`: `<set>_object_id`, `<set>_z` для `valid`, `north`, `nb`,
  `d2`…`d5`, `h6`.
- `phosphoros/out[/<вариант>]/predictions/<SET>.parquet`: `object_id`, `ppub_median`, `ppub_lo68`,
  `ppub_hi68`, `ppub_mode`, `input_status`, `has_prediction`.

## Откуда берутся числа отчёта

Определения: δ = (z_phot − z)/(1 + z), смещение = median δ, NMAD = 1,4826 · median|δ − median δ|,
доля выбросов = доля |δ| ≥ 0,15 (`lib/metrics/metrics.py`; `lib/metrics/bootstrap.photoz_stats`
вызывает эти же функции). Предсказание нашей модели — среднее z по 6 сидам P_final.

Сравниваются наша модель, Phosphoros, Phosphoros без N (`north_zpf`), PHZ Euclid (`PHZ_MEDIAN`,
отрицательные значения отбрасываются) и классические модели. Выборки собирает
`lib/dataloaders/euclid_report.py`:

- `north_frame` — общие объекты N-A + N-B из `labelled_n_table.parquet`, где конечны z и все
  четыре метода: 36 584. Из неё `fig_scatter_N`, `fig_modes`, `fig_bins` и `N_common`.
- `hetdex_frame` — H6, страта `oii`, конечны z, модель, Phosphoros и PHZ, z > 0: 12 859.
- `sources_frame` — N, все источники: 36 584 строки N, D2–D4 (DESI), D5 (NEP-Wide) и HETDEX.
  Остаются строки, где конечны z, все методы и все классические модели, и z > 0: 53 328, из них
  DESI 39 721, NEP-Wide 748, HETDEX 12 859, z ≥ 1 8 802. Из неё `N_sources` в `facts.json`,
  `fig_scatter_Nall` и `fig_baselines`.
- `zval_inputs` — валидация F+S: 1 071 объект из `P_final_<сид>.json` → `zval`. Каждый метод
  считается на своих конечных строках (у PHZ Euclid их 1 063), в таблице стоит n = 1 071.
- `classical_frames` — блок `classical` в `facts.json`: N (все, VIS ≥ 22,5, z ≥ 1), D2–D5,
  F VALID и HETDEX.

Метрики считает `lib/metrics/euclid_eval.py`:

- `source_metrics` → `N_sources`: NMAD, выбросы, смещение по выборкам ALL, ALL_z1, DESI,
  NEP-Wide, HETDEX; ΔNMAD = NMAD(наша) − NMAD(метод) с 95 % ДИ;
- `method_block` → `classical`;
- `zval_metrics` → `VAL`.

ДИ в таблицах — бутстрап по тайлам (`lib/metrics/bootstrap.tile_boot`): 2000 повторов, сид
20261105, ансамбль из 6 сидов фиксирован. В `results_final_vs_{pub,zpf}.json` тайлы и сиды
пересэмплируются вместе (`seed_tile_boot`, 2000 повторов, сид 20261104). Для каждого набора там
три варианта: модель, `+ppub` (среднее log(1+z) модели и Phosphoros с весом 0,5) и `+ppub_ivw`
(веса обратны дисперсиям). Наборы: F_VALID, N_all, N-A, N-B, N_vis_*, N_z_*, HO_S, HO_F_MUSE,
D2–D5. В `results_h6_P_final.json` для HETDEX [OII] `arm_boot` тоже пересэмплирует тайлы и
сиды (2000 повторов, сид 20261208) и сравнивает модель и её гибрид с Phosphoros, Phosphoros без
N и PHZ Euclid.

Откуда берутся числа таблиц:

- `tables/metrics.tex`: строка «F+S: валидация» — `facts.json` → `VAL`; строки N —
  `N_sources.{ALL, ALL_z1, DESI, NEP-Wide, HETDEX}`, колонки `ml`, `ppub`, `poff`.
- `tables/metrics_delta.tex`: `N_sources.<выборка>.d_vs_ppub` и `d_vs_poff`.
- `tables/classical.tex`: `N_sources.{ALL, DESI, NEP-Wide, HETDEX}.<метод>` и
  `N_sources.ALL.d_vs_<метод>`.

Например, на N по всем источникам NMAD 0,0308 у нашей модели, 0,0370 у Phosphoros и 0,0452 у
PHZ Euclid; ΔNMAD относительно Phosphoros −0,0062 [−0,0069; −0,0055]. Значения исходного прогона
заморожены в `reports/weekly/2026-11-06/` (`facts.json`, `metrics/`, `tables/`,
`figures/`).

## Ограничения

- Число «без перевода фотометрии NMAD 0,057 вместо 0,026» в отчётах получено отдельным
  вариантом модели без перевода F/S в полосы UNIONS. Этот вариант в MR не входит, и число
  этим кодом не воспроизводится.
- Сверены с исходным прогоном: загрузка весов и прогноз, разбиение на обучение и валидацию,
  обучающие массивы, перевод фотометрии, классические модели и этап отчёта (таблицы, числа,
  рисунки). Инструменты сборки данных (пакеты, отбор 1 млн объектов, вырезки, индекс, MER-фотометрия,
  подготовка и постобработка Phosphoros, метки) перенесены по исходному коду, но заново
  не прогонялись.
- На замороженных предсказаниях и входах исходного прогона отчёт воспроизводит
  `tables/*.tex` побайтно, все числа `facts.json` и `metrics/*.json` совпадают точно, рисунки
  совпадают попиксельно (растр 100 dpi). Отличается только состав ключей:
  - в `facts.json` нет HistGradBoost (`hgb`), блоков `forest` и `ablation` (их рисунков нет в
    отчёте); `model_args` содержит только ключи метаданных P_final; вместо `store_v1` в
    `store` записаны счётчики, посчитанные по `rows.npz`;
  - в `results_final_vs_*.json` нет сравнения `vs_phosphoros` со старой колонкой `phosphoros`
    таблицы меток N: этот прогон Phosphoros в MR не строится;
  - в `results_h6_P_final.json` нет плеч более ранней модели `R6s_B0` и поля `note`. Плечи
    P_final от этого не меняются.
- Обучение на GPU недетерминировано: включены `cudnn.benchmark` и TF32. Повторный прогон
  совпадает с исходными предсказаниями только с точностью до шума. Для замороженных весов при
  сравнении CPU с GPU медиана |Δ log(1+z)| составляет 4·10⁻⁵–7·10⁻⁵ в зависимости от набора,
  у отдельных объектов с двугорбым распределением по бинам |Δz| доходит до 0,13, а NMAD одного
  сида сдвигается до 3·10⁻⁴ (например, F VALID 0,0202 на CPU против 0,0201 на GPU). На CPU
  повторный прогон совпадает с исходным кодом побитово.
- Скачивание вырезок по объектам (`products/`), реестр объектов F/S, исходная 14-полосная
  фотометрия пакетов и внешние наборы меток (DESI, SDSS, MUSE, D2–D5) в этот MR не входят: это
  входы, перечисленные в таблице выше.
- Phosphoros запускался на Mac (osx-arm64, 6 потоков). Остальная цепочка выполнялась на
  Linux-сервере с GPU.
