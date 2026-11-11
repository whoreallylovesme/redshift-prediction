# Euclid Q1 EDF-N: выгрузка каталогов и нарезка штампов

Конвейер подготовки данных Euclid Q1 (EDF North) для CapsNet photo-z.
Все продукты — открытый IRSA, регистрация не нужна.

## Продукты

| Роль | Продукт IRSA | Тайлов EDF-N |
|---|---|---|
| метки (гризм-z) | `catalogs/SPE_PF_OUTPUT_CATALOG` | 103 (21 без спектроскопии) |
| фотометрия | `catalogs/MER_FINAL_CATALOG` | 124 |
| photo-z бейзлайн | `catalogs/PHZ_PF_OUTPUT_CATALOG` | 111 (13 без продукта) |

Список тайлов EDF-N (124, Dec +63..+69): `edf_n_tiles.json`.

## Порядок

1. `download_q1.sh <spe|mer|phz> <tiles.txt> [ssh-host]` — параллельная выгрузка
   каталогов; при указании ssh-host работает на удалённой машине-прокси
   (у нас nl-VPS: к IRSA оттуда быстрее, потом rsync на локальную машину).
2. `merge_catalogs.py` — сборка таблицы меток: SPE (кат по качеству) ×
   MER (координаты, морфология) × PHZ (UNIF-потоки, PHZ_MEDIAN) по OBJECT_ID.
3. `mosaic_cut.sh <tile> <objects.parquet>` + `cut_stamps.py` — выгрузка
   мозаик BGSUB (VIS, Y, J, H — одна пиксельная сетка) и нарезка штампов
   64×64×4 в один npz на тайл.
4. Обучение: `uv run python runtime/train.py --config-dir configs/nn
   capsnet_euclid` (см. `configs/nn/capsnet_euclid.yaml`).

## Известные грабли

- имена PHZ-файлов не содержат тайл — сопоставление с тайлами только через
  листинг директории тайла или по Content-Length;
- у части тайлов нет SPE (21) или PHZ (13) продуктов — стабильное NO-FILE;
- каталоги клонировать в синк-папку Яндекс.Диска через `cp -c` (APFS-клоны,
  ноль байт) — приложение Диска льёт автономно.
