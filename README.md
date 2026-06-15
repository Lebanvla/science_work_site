# Road Segmentation App

Веб-приложение на FastAPI для асинхронной сегментации дорожной поверхности на изображениях. Пользователь загружает изображение через веб-форму, API сохраняет файл и ставит задачу в Redis, отдельный worker-процесс выполняет инференс модели SegFormer и сохраняет изображение с полупрозрачной оранжевой маской класса `road`.

## Что делает проект

1. Принимает изображения `.jpg`, `.jpeg`, `.png`, `.webp`.
2. Считает SHA256 содержимого файла и использует его как `picture_hash`.
3. Сохраняет исходное изображение в `app/storage/input/`.
4. Кладёт задачу в Redis-очередь `segmentation_queue`.
5. Worker забирает задачу через `BLPOP`, запускает модель `nvidia/segformer-b0-finetuned-cityscapes-768-768` и сохраняет результат в `app/storage/output/`.
6. API позволяет опрашивать статус обработки по `GET /get/{picture_hash}`.
7. Через 1 час worker удаляет исходное изображение и результат. Redis-ключ результата также хранится с TTL 3600 секунд.

## Архитектура

```text
Browser
  |
  | POST /predict
  v
FastAPI API
  |  сохраняет input-файл
  |  RPUSH segmentation_queue
  v
Redis
  |
  | BLPOP segmentation_queue
  v
Worker
  |  модель загружается один раз при старте процесса
  |  инференс SegFormer
  |  сохранение output-файла
  |  SETEX result:{picture_hash} output_hash 3600
  v
Browser
  |
  | GET /get/{picture_hash} каждые 2 секунды
  v
/static/output/<output_hash>.png
```

FastAPI не загружает модель и не выполняет долгий инференс внутри endpoint. Инференс полностью вынесен в отдельный worker.

## Структура проекта

```text
road-segmentation-app/
    app/
        main.py
        model_runner.py
        worker.py
        redis_client.py
        config.py
        templates/
            index.html
        static/
            output/
        storage/
            input/
            output/
    requirements.txt
    Dockerfile
    docker-compose.yml
    .dockerignore
    README.md
```

При запуске через Docker Compose используется volume:

```text
./storage:/app/app/storage
```

То есть файлы ввода и вывода будут храниться на хосте в папке `./storage`, а внутри контейнеров будут доступны как `/app/app/storage`.

## Endpoints

### `GET /`

Возвращает HTML-страницу с формой загрузки изображения. JavaScript отправляет файл на `/predict`, получает `picture_hash` и каждые 2 секунды опрашивает `/get/{picture_hash}`.

### `POST /predict`

Принимает файл изображения через multipart form-data поле `file`.

Успешный ответ:

```json
{
  "status": "queued",
  "picture_hash": "..."
}
```

Если файл пустой, слишком большой, имеет неподдерживаемое расширение или не является изображением, API вернёт ошибку `400`.

### `GET /get/{picture_hash}`

Если результат ещё не готов:

```json
{
  "status": "processing"
}
```

Если результат готов:

```json
{
  "status": "ready",
  "url": "/static/output/<output_hash>.png"
}
```

Если worker получил ошибку обработки:

```json
{
  "status": "error",
  "message": "..."
}
```

## Запуск через Docker

Соберите образы:

```bash
docker compose build
```

Запустите сервисы:

```bash
docker compose up -d
```

Откройте приложение:

```text
http://localhost:8000
```

Первый запуск worker может занять больше времени, потому что модель и processor будут скачаны из Hugging Face. Контейнеру нужен доступ в интернет.

## Проверка логов

Логи API:

```bash
docker compose logs -f api
```

Логи worker:

```bash
docker compose logs -f worker
```

Логи Redis:

```bash
docker compose logs -f redis
```

## Остановка проекта

```bash
docker compose down
```

Если хотите удалить сохранённые изображения на хосте:

```bash
rm -rf storage
```

## Переменные окружения

Основные переменные уже заданы в `docker-compose.yml`:

```text
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
MODEL_NAME=nvidia/segformer-b0-finetuned-cityscapes-768-768
```

Дополнительно можно настроить:

```text
ROAD_CLASS_NAME=road
ROAD_CLASS_INDEX=0
RESULT_TTL_SECONDS=3600
CLEANUP_DELAY_SECONDS=3600
MAX_UPLOAD_SIZE_BYTES=26214400
```

`ROAD_CLASS_INDEX` полезен, если вы подключите модель без корректной таблицы `id2label`.

## Примечание по PyTorch CPU wheel

Установка `torch` в Docker может быть тяжёлой. Для CPU-инференса можно заменить установку PyTorch на официальный CPU wheel, например адаптировать Dockerfile так:

```dockerfile
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
```

В таком случае из `requirements.txt` лучше убрать `torch` и `torchvision`, чтобы они не переустанавливались из обычного PyPI.

## Как заменить teacher-модель на student-модель

FastAPI и Redis-логика не зависят от конкретной нейросети. Модельный слой находится в `app/model_runner.py`.

Есть базовый интерфейс:

```python
class SegmentationModel:
    def predict(self, image_path: str) -> np.ndarray:
        ...
```

Текущая реализация:

```python
class SegFormerCityscapesModel(SegmentationModel):
    ...
```

Чтобы подключить свою student-модель:

1. Создайте новый класс, например `StudentRoadSegmentationModel(SegmentationModel)`.
2. Реализуйте метод `predict(self, image_path: str) -> np.ndarray`, который возвращает бинарную маску дороги `uint8` размера исходного изображения.
3. Измените функцию `load_model()` так, чтобы она создавала ваш класс вместо `SegFormerCityscapesModel`.
4. Остальной код `main.py`, `worker.py`, Redis-очередь и API менять не нужно.

Пример:

```python
class StudentRoadSegmentationModel(SegmentationModel):
    def __init__(self, weights_path: str):
        self.model = load_my_student_model(weights_path)

    def predict(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        mask = run_student_inference(self.model, image)
        return mask.astype(np.uint8)


def load_model() -> SegmentationModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = StudentRoadSegmentationModel("/app/models/student.pt")
    return _MODEL
```

Главное условие: `predict()` должен возвращать бинарную маску дороги, где пиксели дороги имеют значение `255`, а остальные — `0`.
