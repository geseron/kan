import datetime
import os

def v_log(message, filename='log.txt'):
    """
    Записывает сообщение в лог‑файл с датой и временем.
    Файл сохраняется в папку ./logs/.
    
    :param message: строка — сообщение для записи
    :param filename: строка — имя файла лога (по умолчанию 'log.txt')
    """
    # Создаём папку logs, если её нет
    os.makedirs('logs', exist_ok=True)
    
    # Формируем полный путь к файлу
    filepath = os.path.join('logs', filename)
    
    # Получаем текущее время
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Записываем в файл
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] {message}\n')
