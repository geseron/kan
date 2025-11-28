import pickle
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any
import os

import pickle
import pyarrow as pa
import pyarrow.parquet as pq
import os
from typing import List, Dict, Any, Optional


def save_experiment_data(
    rows: List[Dict[str, Any]],
    base_path: str,
    formats: Optional[List[str]] = ["pkl"],  # Новый параметр
    protocol: int = pickle.HIGHEST_PROTOCOL,
    parquet_compression: str = "snappy"
) -> dict:
    """
    Сохраняет список словарей (rows) в указанные форматы.


    Параметры:
    -----------
    rows : List[Dict[str, Any]]
        Список словарей с данными экспериментов.
    base_path : str
        Базовый путь без расширения (например, "results/exp_1").
    formats : List[str], optional
        Список форматов для сохранения. Допустимые значения:
        - "pkl"  (pickle)
        - "pq"   (parquet)
        Если None — сохраняются оба формата.
    protocol : int, optional
        Протокол pickle (по умолчанию — максимальный для текущей версии Python).
        Для совместимости с Python 2.x укажите protocol=2.
    parquet_compression : str, optional
        Алгоритм сжатия для Parquet (по умолчанию "snappy").
        Другие варианты: "gzip", "zstd", "none".

    Возвращает:
    ----------
    dict с путями к сохранённым файлам и количеством записей.
    """
    if not rows:
        raise ValueError("Список rows пуст!")

    # Определяем форматы для сохранения
    if formats is None:
        formats = ["pkl", "pq"]  # По умолчанию — оба формата
    else:
        # Проверяем корректность форматов
        valid_formats = {"pkl", "pq"}
        for fmt in formats:
            if fmt not in valid_formats:
                raise ValueError(f"Неизвестный формат: {fmt}. Допустимые: {valid_formats}")

    result = {
        "num_records": len(rows),
        "protocol_used": protocol,
        "compression": parquet_compression,
        "saved_formats": {}
    }

    # Сохранение в pickle
    if "pkl" in formats:
        pickle_path = f"{base_path}.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(rows, f, protocol=protocol)
        result["saved_formats"]["pkl"] = os.path.abspath(pickle_path)


    # Сохранение в parquet
    if "pq" in formats:
        parquet_path = f"{base_path}.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(
            table,
            parquet_path,
            compression=parquet_compression
        )
        result["saved_formats"]["pq"] = os.path.abspath(parquet_path)

    return result