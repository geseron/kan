import itertools
import pandas as pd
import numpy as np
import torch
#from kan import KAN
from kan.utils import create_dataset_from_data, ex_round
from joblib import Parallel, delayed
import multiprocessing
import time
from logs import *
from save import *
import math
from tqdm import tqdm


# === Функции ===
def f1(x): return np.sin(np.pi * x) + x**2
def f2(x): return np.exp(-x**2)
def f3(x): return x * np.sin(5 * x)
def f4(x): return np.sin(np.pi * x[:, 0]) + x[:, 1]**2
def f5(x): return np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1]**2)

FUNCTIONS = [
    (f1, 1, "sin(pi*x) + x^2", "D1N1"),
    (f2, 1, "exp(-x^2)", "D1N2"),
    (f3, 1, "x * sin(5*x)", "D1N3"),
    (f4, 2, "sin(pi*x) + y^2", "D2N1"),
    (f5, 2, "exp(sin(pi*x) + y^2)", "D2N2"),
]
lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']

# n_samples_list = [100, 200, 400, 500]
# noise_levels = [0, 0.05, 0.1, 0.3, 0.5]
# #gap_ranges = [None, [-1, 1], [0.5, 1]]
# gap_ranges = [None]


# width_list = [[5, 1],[4, 4, 4, 1],[15, 1],[5, 5, 1],[10, 6, 3, 1]]
# k_list = [1, 2, 3, 5]
# grid_list = [3, 5]

n_samples_list = [100]
noise_levels = [0, 0.05]
#gap_ranges = [None, [-1, 1], [0.5, 1]]
gap_ranges = [None]


width_list = [[5, 1],[4, 4, 4, 1]]
k_list = [1]
grid_list = [3]


# === Новые настройки ===
n_repeats = 5  # Количество запусков на одну комбинацию
x_true = np.linspace(-2, 2, 500)

# === Вспомогательные утилиты ===
def extract_loss_history(fit_result, key):
    if fit_result is None or key not in fit_result:
        return []
    val = fit_result[key]
    if isinstance(val, (list, np.ndarray)):
        return str([float(t) for t in val])
    elif isinstance(val, (float, int)):
        return [val]
    return []

def stats(model, dataset, func, n_var, x_true):
    x_true_clear = x_true.copy()
    if n_var == 2:
        x_true_clear = np.column_stack([x_true, x_true])
    y_true_clear = func(x_true_clear)

    with torch.no_grad():
        device = model.device
        y_pred = model(dataset['test_input'])
        y_true = dataset['test_label']
        y_true_pred = model(
            torch.tensor(x_true_clear, dtype=torch.float32, device=device).reshape((len(x_true), n_var))
        )

    y_pred_np = y_pred.cpu().numpy().flatten()
    y_true_np = y_true.cpu().numpy().flatten()
    y_true_pred_np = y_true_pred.cpu().numpy().flatten()

    residuals = y_true_np - y_pred_np
    mse = np.mean(residuals ** 2)
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))

    residuals_true = y_true_clear - y_true_pred_np
    mse_true = np.mean(residuals_true ** 2)
    mae_true = np.mean(np.abs(residuals_true))
    ss_res_true = np.sum(residuals_true ** 2)
    ss_tot_true = np.sum((y_true_clear - np.mean(y_true_clear)) ** 2)
    r2_true = 1 - (ss_res_true / (ss_tot_true + 1e-12))

    return {
        'test_residuals': residuals,
        'test_mse': mse,
        'test_mae': mae,
        'test_r2': r2,
        'true_residuals': residuals_true,
        'true_mse': mse_true,
        'true_mae': mae_true,
        'true_r2': r2_true,
    }

def run_single_experiment(args):
    (
        func, n_var, func_id, id_name,
        gap_range, n_samples, noise_level,
        width_tail, k, grid
    ) = args
    import kan

    # Полное отключение сохранения
    def dummy_log_history(self, method_name): pass
    def dummy_saveckpt(self, path): pass
    kan.KAN.log_history = dummy_log_history
    kan.KAN.saveckpt = dummy_saveckpt
    
    seed = np.random.randint(0, 2**30)
    gap_str = f"gap{gap_range[0]}_{gap_range[1]}" if gap_range else "nogap"
    width_current = [n_var] + width_tail
    filename = f'datasets/data_{id_name}_noise{noise_level}_{gap_str}_n{n_samples}.csv'

    try:
        df = pd.read_csv(filename, index_col=False)
    except FileNotFoundError:
        return None

    device = torch.device('cpu')
    if n_var == 1:
        x = torch.tensor(df['x1'].values, dtype=torch.float32).reshape(-1, 1)
    else:
        x = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
    y = torch.tensor(df['y'].values, dtype=torch.float32).reshape(-1, 1)
    dataset = create_dataset_from_data(x, y)

    model = kan.KAN(width=width_current.copy(), grid=grid, k=k, seed=seed, device=device, auto_save=False)

    # === Функция проверки NaN в истории лосса ===
    def has_nan(history):
        if history is None:
            return True
        if isinstance(history, (list, np.ndarray)):
            return any(np.isnan(v) for v in history if v is not None)
        return np.isnan(history)

    # Этап 1: до prune
    res_before = model.fit(dataset, opt="LBFGS", steps=20, lamb=0.001, update_grid=False)
    if res_before is None or has_nan(res_before['train_loss']) or has_nan(res_before['test_loss']) or has_nan(res_before['reg']):
        return None
    stats_before = stats(model, dataset, func, n_var, x_true)

    # Этап 2: после prune
    try:
        model = model.prune()
    except Exception as e:
        if "non-empty TensorList" in str(e) or "in_dim" in str(e):
            return None
        else:
            raise e
    res_after_prune = model.fit(dataset, opt="LBFGS", steps=20, update_grid=False)
    if res_after_prune is None or has_nan(res_before['train_loss']) or has_nan(res_before['test_loss']) or has_nan(res_before['reg']):
        return None
    stats_after_prune = stats(model, dataset, func, n_var, x_true)

    # Этап 3: после refine
    model = model.refine(10)
    res_after_refine = model.fit(dataset, opt="LBFGS", steps=20, update_grid=False)
    if res_after_refine is None or has_nan(res_before['train_loss']) or has_nan(res_before['test_loss']) or has_nan(res_before['reg']):
        return None
    stats_after_refine = stats(model, dataset, func, n_var, x_true)

    # Этап 4: symbolic
    model.auto_symbolic(lib=lib, verbose=False)
    res_after_symbolic = model.fit(dataset, opt="LBFGS", steps=20, update_grid=False)
    if res_after_symbolic is None or has_nan(res_before['train_loss']) or has_nan(res_before['test_loss']) or has_nan(res_before['reg']):
        return None
    stats_after_symbolic = stats(model, dataset, func, n_var, x_true)

    formula = str(ex_round(model.symbolic_formula()[0][0], 4))

    row = {
        'func_id': id_name,
        'noise_level': noise_level,
        'gap_str': gap_str,
        'n_samples': n_samples,
        'width': str(width_current),
        'k': k,
        'grid': grid,
        'seed': seed,
        'formula': formula,
        'loss_train_history_before_prune': extract_loss_history(res_before, 'train_loss'),
        'loss_test_history_before_prune': extract_loss_history(res_before, 'test_loss'),
        'reg_history_before_prune': extract_loss_history(res_before, 'reg'),
        'loss_train_history_after_prune': extract_loss_history(res_after_prune, 'train_loss'),
        'loss_test_history_after_prune': extract_loss_history(res_after_prune, 'test_loss'),
        'reg_history_after_prune': extract_loss_history(res_after_prune, 'reg'),
        'loss_train_history_after_refine': extract_loss_history(res_after_refine, 'train_loss'),
        'loss_test_history_after_refine': extract_loss_history(res_after_refine, 'test_loss'),
        'reg_history_after_refine': extract_loss_history(res_after_refine, 'reg'),
        'loss_train_history_after_symbolic': extract_loss_history(res_after_symbolic, 'train_loss'),
        'loss_test_history_after_symbolic': extract_loss_history(res_after_symbolic, 'test_loss'),
        'reg_history_after_symbolic': extract_loss_history(res_after_symbolic, 'reg'),
    }

    for stage_name, stat_dict in [
        ('before_prune', stats_before),
        ('after_prune', stats_after_prune),
        ('after_refine', stats_after_refine),
        ('after_symbolic', stats_after_symbolic),
    ]:
        for key in ['test_residuals', 'test_mse', 'test_mae', 'test_r2',
                    'true_residuals', 'true_mse', 'true_mae', 'true_r2']:
            row[f'{key}_{stage_name}'] = stat_dict[key]

    return row

# === Столбцы ===
base_cols = [
    'func_id', 'noise_level', 'gap_str', 'n_samples',
    'width', 'k', 'grid', 'seed', 'formula'  # добавили seed
]
stages = ['before_prune', 'after_prune', 'after_refine', 'after_symbolic']
loss_cols = []
metric_cols = []
for stage in stages:
    loss_cols += [
        f'loss_train_history_{stage}',
        f'loss_test_history_{stage}',
        f'reg_history_{stage}',
    ]
    metric_cols += [
        f'test_residuals_{stage}',
        f'test_mse_{stage}',
        f'test_mae_{stage}',
        f'test_r2_{stage}',
        f'true_residuals_{stage}',
        f'true_mse_{stage}',
        f'true_mae_{stage}',
        f'true_r2_{stage}',
    ]
all_columns = base_cols + loss_cols + metric_cols


from joblib import Parallel, delayed
from tqdm import tqdm
import time
import os
import math

# Параметры
MAX_RETRIES = 3
TIMEOUT_SECONDS = 200
BATCH_SIZE = 36
n_jobs = 12

# Путь к папке с результатами
RESULTS_DIR = "results"

def is_batch_completed(id_name, batch_num):
    """
    Проверяет, выполнена ли пачка
    """
    parquet_path = f"{RESULTS_DIR}/results_{id_name}_{batch_num}.pkl"
    return os.path.isfile(parquet_path)


for func_idx, (func, n_var, name, id_name) in enumerate(FUNCTIONS):
    print(f"\n🚀 Запуск экспериментов для функции: {id_name}")
    v_log(f'Функция {id_name} начата')

    # Генерация всех задач (как раньше)
    tasks = []
    for gap_range in gap_ranges:
        for n_samples in n_samples_list:
            for noise_level in noise_levels:
                for width in width_list:
                    for k in k_list:
                        for grid in grid_list:
                            for i in range(n_repeats):
                                tasks.append((
                    func, n_var, func_idx, id_name,
                    gap_range, n_samples, noise_level,
                    width, k, grid
                ))

    total = len(tasks)
    print(f"Всего задач для {id_name}: {total}")
    n_batches = math.ceil(total / BATCH_SIZE)
    print(f"Разбито на {n_batches} пачек по {BATCH_SIZE} задач")


    # Обработка пачками
    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_tasks = tasks[start_idx:end_idx]
        batch_num = batch_idx + 1
        batch_total = len(batch_tasks)

        # Проверяем, выполнена ли эта пачка
        if is_batch_completed(id_name, batch_num):
            print(f"✅ Пачка {batch_num}/{n_batches} уже выполнена (файл существует). Пропускаем.")
            v_log(f'{id_name} Пачка {batch_num} пропущена: файл уже есть')
            continue

        print(f"Обработка пачки {batch_num}/{n_batches} ({batch_total} задач)")


        # Цикл попыток с перезапуском при таймауте
        batch_results = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"!Попытка {attempt}/{MAX_RETRIES} для пачки {batch_num}")
                batch_results = Parallel(n_jobs=n_jobs, timeout=TIMEOUT_SECONDS, verbose=5)(
                    delayed(run_single_experiment)(task)
                    for task in tqdm(batch_tasks, desc=f"Пачка {batch_num} {id_name}", total=batch_total)
                )
                v_log(f'{id_name} Пачка {batch_num} завершена за {attempt} попытку. Результат: {len(batch_results)} записей')
                break  # Успех — выходим из цикла попыток

            except multiprocessing.TimeoutError:
            # Основной случай: таймаут joblib
                print(f"!⚠️ Таймаут! Пачка {batch_num}, попытка {attempt}")
                v_log(f'{id_name} Пачка {batch_num}, попытка {attempt} — TimeoutError')
                if attempt < MAX_RETRIES:
                    print(f"Перезапуск пачки {batch_num} (попытка {attempt + 1})...")
                else:
                    print(f"❌ Все {MAX_RETRIES} попыток исчерпаны для пачки {batch_num}. Пропускаем.")
                    v_log(f'{id_name} Пачка {batch_num}: все попытки неудачны, данных нет')
                    batch_results = []  # Пустой результат для дальнейшей обработки



        # Если результатов нет (все попытки провалились), пропускаем сохранение
        if batch_results is None or not batch_results:
            print(f"⚠️ Пачка {batch_num}: нет валидных результатов после {MAX_RETRIES} попыток")
            v_log(f'{id_name} Пачка {batch_num}: нет данных для сохранения')
            continue

        # Фильтрация None
        batch_rows = [r for r in batch_results if r is not None]
        v_log(f'{id_name} Пачка {batch_num} Фильтрация завершена. Строк: {len(batch_rows)}')


        if not batch_rows:
            print(f"⚠️ Пачка {batch_num}: нет валидных результатов после фильтрации")
            v_log(f'{id_name} Пачка {batch_num}: нет данных для сохранения после фильтрации')
            continue

        # Сохранение пачки
        batch_base_path = f"{RESULTS_DIR}/results_{id_name}_{batch_num}"
        save_result = save_experiment_data(
            rows=batch_rows,
            base_path=batch_base_path,
            protocol=4,
            parquet_compression="snappy"
        )
        v_log(f'{id_name} Пачка {batch_num} сохранена: {save_result}')

        # Опционально: сохранение CSV
        # df_batch = pd.DataFrame(batch_rows)
        # csv_file = f"{RESULTS_DIR}/results_{id_name}_{batch_num}.csv"
        # df_batch.to_csv(csv_file, index=False, sep=';')
        # print(f"✅ Сохранено: {csv_file} ({len(batch_rows)} экспериментов)")
        # v_log(f'{id_name} Пачка {batch_num} CSV сохранён: {csv_file}')

print("Все функции обработаны!")
