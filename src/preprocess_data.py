import argparse
import json
import os

import pandas as pd
from PIL import Image, UnidentifiedImageError


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

REMOVED_CLASS = 18
OLD_TO_NEW_CLASS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    19: 18,
}

# Эти heuristics-датасеты можно добавлять в train по отдельности
HEURISTICS = {
    "cabinet": {
        "csv": "heuristics_cabinet.csv",
        "result": 5,
        "label": "кабинет",
    },
    "detskaya": {
        "csv": "heuristics_detskaya.csv",
        "result": 6,
        "label": "детская",
    },
    "dressing_room": {
        "csv": "heuristics_dressing_room.csv",
        "result": 11,
        "label": "гардеробная / кладовая / постирочная",
    },
}
RECOMMENDED_HEURISTICS = ["cabinet", "dressing_room"]


def parse_args():
    """Читает аргументы командной строки для подготовки данных
    Returns:
        Разобранные аргументы командной строки
    """
    parser = argparse.ArgumentParser(description="Подготовить raw CSV-файлы для обучения модели")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Проверять только наличие файлов, не открывая изображения через PIL",
    )
    parser.add_argument(
        "--include-heuristics",
        default="none",
        help="none, recommended, all или список через запятую: cabinet,dressing_room",
    )
    parser.add_argument(
        "--max-heuristics-per-source",
        type=int,
        default=None,
        help="Дополнительный лимит строк из каждого heuristics-датасета",
    )
    parser.add_argument(
        "--heuristics-seed",
        type=int,
        default=42,
        help="Seed для случайного выбора строк из heuristics",
    )
    return parser.parse_args()


def is_valid_image(image_path):
    """Проверяет, что файл изображения открывается через PIL
    Args:
        image_path: Путь к файлу изображения

    Returns:
        True, если PIL успешно проверил изображение, иначе False
    """
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def choose_heuristics(value):
    """Возвращает список heuristics-датасетов из CLI-аргумента"""
    value = value.strip()
    if value == "none":
        return []
    if value == "recommended":
        return RECOMMENDED_HEURISTICS
    if value == "all":
        return list(HEURISTICS)

    names = [name.strip() for name in value.split(",") if name.strip()]
    wrong_names = [name for name in names if name not in HEURISTICS]
    if wrong_names:
        raise ValueError(f"Неизвестные heuristics-датасеты: {', '.join(wrong_names)}")
    return names


def check_images(df, image_root, image_ext, verify_images):
    """Проверяет наличие и валидность локальных изображений

    Args:
        df: DataFrame с колонкой `image_id_ext`
        image_root: Папка с изображениями split
        image_ext: Расширение файла изображения
        verify_images: Нужно ли открывать изображения через PIL

    Returns:
        Тот же DataFrame с колонками `image_exists`, `image_is_valid` и `can_predict`
    """
    image_exists = []
    image_is_valid = []

    for image_id in df["image_id_ext"]:
        if pd.isna(image_id):
            image_exists.append(False)
            image_is_valid.append(False)
            continue

        image_path = os.path.join(image_root, f"{image_id}{image_ext}")
        exists = os.path.exists(image_path)
        image_exists.append(exists)

        if not exists:
            image_is_valid.append(False)
        elif verify_images:
            image_is_valid.append(is_valid_image(image_path))
        else:
            image_is_valid.append(True)

    df["image_exists"] = image_exists
    df["image_is_valid"] = image_is_valid
    df["can_predict"] = df["image_exists"] & df["image_is_valid"]
    return df


def add_image_path(df, image_folder, source, is_auxiliary):
    """Добавляет признаки, по которым Dataset найдёт изображение"""
    df["source"] = source
    df["is_auxiliary"] = is_auxiliary
    df["image_path"] = df["image_id_ext"].map(lambda image_id: f"{image_folder}/{image_id}.jpg")
    return df


def normalize_title(title):
    """Упрощает title для поиска дублей"""
    if pd.isna(title):
        return None
    return " ".join(str(title).lower().split())


def preprocess_train_val(split, raw_dir, processed_dir, image_ext, verify_images):
    """Очищает train или val и сохраняет CSV для обучения

    Args:
        split: Название split, `train` или `val`
        raw_dir: Папка с raw-данными
        processed_dir: Папка для обработанных CSV
        image_ext: Расширение файла изображения
        verify_images: Нужно ли открывать изображения через PIL

    Returns:
        Количество строк до и после обработки
    """
    csv_path = os.path.join(raw_dir, f"{split}_df.csv")
    image_root = os.path.join(raw_dir, f"{split}_images")
    df = pd.read_csv(csv_path)
    rows_before = len(df)

    df = df[["image_id_ext", "image", "result", "label"]].copy()
    df["title"] = None

    df["image_id_ext"] = df["image_id_ext"].astype(str).str.strip()
    df["image_id_ext"] = df["image_id_ext"].replace({"": None, "nan": None})
    df = df.dropna(subset=["image_id_ext"])

    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df = df.dropna(subset=["result"])
    df["result"] = df["result"].astype(int)

    df = df[df["result"] != REMOVED_CLASS]
    df = df[df["result"].isin(OLD_TO_NEW_CLASS)]

    df = check_images(df, image_root, image_ext, verify_images)
    df = df[df["can_predict"]].copy()
    df = df.drop(columns=["image_exists", "image_is_valid", "can_predict"])

    # Сначала убираем полностью одинаковые строки, потом повторы одного изображения
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["image_id_ext"], keep="first")

    # После удаления класса 18 переносим старый класс 19 в новый id 18
    df["result"] = df["result"].replace(OLD_TO_NEW_CLASS)
    df = add_image_path(df, f"{split}_images", split, False)
    df = df.reset_index(drop=True)

    output_csv = os.path.join(processed_dir, f"{split}_df.csv")
    df.to_csv(output_csv, index=False)
    return rows_before, len(df)


def read_heuristic_dataset(name, raw_dir, image_ext, verify_images):
    """Читает один heuristics-датасет и приводит его к схеме train"""
    config = HEURISTICS[name]
    csv_path = os.path.join(raw_dir, config["csv"])
    image_root = os.path.join(raw_dir, "heuristics_images")
    df = pd.read_csv(csv_path)
    rows_before = len(df)

    df = df[["image_id_ext", "image", "title"]].copy()
    df["image_id_ext"] = df["image_id_ext"].astype(str).str.strip()
    df["image_id_ext"] = df["image_id_ext"].replace({"": None, "nan": None})
    df = df.dropna(subset=["image_id_ext"])

    df["result"] = config["result"]
    df["label"] = config["label"]

    df = check_images(df, image_root, image_ext, verify_images)
    df = df[df["can_predict"]].copy()
    df = df.drop(columns=["image_exists", "image_is_valid", "can_predict"])

    # Один и тот же файл может попасть в heuristics несколько раз
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["image_id_ext"], keep="first")
    rows_after_image_dedup = len(df)

    # В heuristics много дублей с разными картинками, но одинаковым title
    df["title_for_dedup"] = df["title"].map(normalize_title)
    df_with_title = df[df["title_for_dedup"].notna()].copy()
    df_without_title = df[df["title_for_dedup"].isna()].copy()
    # Сортировка делает выбор дубля по title воспроизводимым
    df_with_title = df_with_title.sort_values("image_id_ext")
    df_with_title = df_with_title.drop_duplicates(subset=["title_for_dedup"], keep="first")
    df = pd.concat([df_with_title, df_without_title], ignore_index=True)
    df = df.drop(columns=["title_for_dedup"])
    rows_after_title_dedup = len(df)

    source = f"heuristics_{name}"
    df = add_image_path(df, "heuristics_images", source, True)
    df = df.reset_index(drop=True)

    stats = {
        "rows_before": rows_before,
        "rows_after_image_dedup": rows_after_image_dedup,
        "rows_after_title_dedup": rows_after_title_dedup,
    }
    return df, stats


def add_heuristics_to_train(train_df, heuristic_names, raw_dir, image_ext, verify_images, max_rows, seed):
    """Добавляет heuristics-датасеты так, чтобы классы дошли до среднего размера"""
    stats = {}
    if not heuristic_names:
        return train_df, stats, None

    # Heuristics нужны только для добора слабых классов до среднего размера
    class_counts = train_df["result"].value_counts()
    if class_counts.empty:
        raise ValueError("после очистки train не осталось строк")
    target_count = round(class_counts.mean())
    frames = [train_df]

    for name in heuristic_names:
        config = HEURISTICS[name]
        class_id = config["result"]
        current_count = int((train_df["result"] == class_id).sum())
        # Если класс уже не меньше среднего, ничего из heuristics не добавляем
        need_count = max(target_count - current_count, 0)

        # Лимит нужен как дополнительный предохранитель от слишком большого добавления
        if max_rows is not None:
            need_count = min(need_count, max_rows)

        heuristic_df, source_stats = read_heuristic_dataset(
            name,
            raw_dir,
            image_ext,
            verify_images,
        )

        if need_count > 0 and len(heuristic_df) > need_count:
            heuristic_df = heuristic_df.sample(n=need_count, random_state=seed)
        else:
            heuristic_df = heuristic_df.head(need_count)

        frames.append(heuristic_df)
        stats[name] = {
            **source_stats,
            "class_id": class_id,
            "class_count_before": current_count,
            "target_count": target_count,
            "added_rows": len(heuristic_df),
        }

    # Основной train стоит первым, чтобы при дублях сохранить исходную разметку
    train_df = pd.concat(frames, ignore_index=True)
    train_df = train_df.drop_duplicates(subset=["image_id_ext"], keep="first")
    train_df = train_df.reset_index(drop=True)
    return train_df, stats, target_count


def preprocess_test(raw_dir, processed_dir, image_ext, verify_images):
    """Готовит test без удаления строк

    Args:
        raw_dir: Папка с raw-данными
        processed_dir: Папка для обработанных CSV
        image_ext: Расширение файла изображения
        verify_images: Нужно ли открывать изображения через PIL

    Returns:
        Количество строк до обработки, после обработки и количество строк для предсказания
    """
    csv_path = os.path.join(raw_dir, "test_df.csv")
    image_root = os.path.join(raw_dir, "test_images")
    df = pd.read_csv(csv_path)
    rows_before = len(df)

    # В test не удаляем строки, чтобы не сломать порядок и количество объектов для проверки
    df = df[["image_id_ext", "image", "item_id"]].copy()
    df["image_id_ext"] = df["image_id_ext"].astype(str).str.strip()
    df["image_id_ext"] = df["image_id_ext"].replace({"": None, "nan": None})
    df = check_images(df, image_root, image_ext, verify_images)
    df = add_image_path(df, "test_images", "test", False)
    df = df.reset_index(drop=True)

    output_csv = os.path.join(processed_dir, "test_df.csv")
    df.to_csv(output_csv, index=False)
    return rows_before, len(df), int(df["can_predict"].sum())


def save_class_mapping(processed_dir):
    """Сохраняет mapping классов после удаления старого класса 18"""
    class_mapping = {
        "target_column": "result",
        "removed_old_classes": [REMOVED_CLASS],
        "old_to_new": {str(old_id): new_id for old_id, new_id in OLD_TO_NEW_CLASS.items()},
        "new_to_old": {str(new_id): old_id for old_id, new_id in OLD_TO_NEW_CLASS.items()},
    }
    class_mapping_path = os.path.join(processed_dir, "class_mapping.json")

    with open(class_mapping_path, "w", encoding="utf-8") as file:
        json.dump(class_mapping, file, indent=2, ensure_ascii=False)

    return class_mapping_path


def main():
    """Запускает подготовку train, val и test split"""
    args = parse_args()
    os.makedirs(args.processed_dir, exist_ok=True)

    train_before, train_after = preprocess_train_val(
        "train",
        args.raw_dir,
        args.processed_dir,
        args.image_ext,
        not args.skip_image_verify,
    )

    train_csv_path = os.path.join(args.processed_dir, "train_df.csv")
    train_df = pd.read_csv(train_csv_path)
    heuristic_names = choose_heuristics(args.include_heuristics)
    train_df, heuristic_stats, target_count = add_heuristics_to_train(
        train_df,
        heuristic_names,
        args.raw_dir,
        args.image_ext,
        not args.skip_image_verify,
        args.max_heuristics_per_source,
        args.heuristics_seed,
    )
    train_df.to_csv(train_csv_path, index=False)

    val_before, val_after = preprocess_train_val(
        "val",
        args.raw_dir,
        args.processed_dir,
        args.image_ext,
        not args.skip_image_verify,
    )
    test_before, test_after, test_can_predict = preprocess_test(
        args.raw_dir,
        args.processed_dir,
        args.image_ext,
        not args.skip_image_verify,
    )

    class_mapping_path = save_class_mapping(args.processed_dir)

    # Manifest помогает понять, с какими heuristics был собран текущий train
    manifest = {
        "include_heuristics": heuristic_names,
        "heuristics_target_count": target_count,
        "heuristics": heuristic_stats,
        "train_rows_before_heuristics": train_after,
        "train_rows_after_heuristics": len(train_df),
    }
    manifest_path = os.path.join(args.processed_dir, "preprocessing_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"Папка с обработанными данными: {args.processed_dir}")
    print(f"Mapping классов: {class_mapping_path}")
    print(f"Manifest preprocessing: {manifest_path}")
    print(f"train: строк до={train_before}, строк после={train_after}")
    print(f"train: строк после heuristics={len(train_df)}")
    print(f"train: включены heuristics={heuristic_names or 'нет'}")
    print(f"train: целевой размер класса для heuristics={target_count or 'нет'}")
    print(f"val: строк до={val_before}, строк после={val_after}")
    print(f"test: строк до={test_before}, строк после={test_after}")
    print(f"test: строк для предсказания={test_can_predict} из {test_after}")


if __name__ == "__main__":
    main()
