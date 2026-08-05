Шаг 2. Скачать образы на компьютере №1
docker pull --platform linux/amd64 \
  paddlepaddle/paddle:3.2.2-gpu-cuda11.8-cudnn8.9

docker pull --platform linux/amd64 python:3.11-slim
docker pull --platform linux/amd64 postgres:16-alpine
docker pull --platform linux/amd64 docker/dockerfile:1.7
Шаг 3. Сохранить GPU-образы
docker save \
  -o fastservice-base-gpu.tar \
  paddlepaddle/paddle:3.2.2-gpu-cuda11.8-cudnn8.9 \
  python:3.11-slim \
  postgres:16-alpine \
  docker/dockerfile:1.7
Контрольная сумма:
sha256sum fastservice-base-gpu.tar \
  > fastservice-base-gpu.tar.sha256
Перенесите оба файла на компьютер №2.
Образ может занимать больше 4 ГБ. Флешка должна быть отформатирована в exFAT или NTFS, а не FAT32.
Шаг 4. Загрузить GPU-образы на компьютере №2
sha256sum -c fastservice-base-gpu.tar.sha256
sudo docker load -i fastservice-base-gpu.tar
Проверить:
sudo docker image inspect \
  paddlepaddle/paddle:3.2.2-gpu-cuda11.8-cudnn8.9

sudo docker image inspect python:3.11-slim
sudo docker image inspect postgres:16-alpine
sudo docker image inspect docker/dockerfile:1.7
Проверить GPU:
nvidia-smi
Также на компьютере №2 должен быть установлен NVIDIA Container Toolkit.
Шаг 5. Собрать GPU-версию
cd /путь/к/fastservice

sudo docker compose \
  --progress plain \
  -f docker-compose.gpu.yml \
  build recognition admin
Шаг 6. Запустить GPU-версию
sudo docker compose \
  -f docker-compose.gpu.yml \
  up -d \
  --no-build \
  --pull never
Проверка:
sudo docker compose -f docker-compose.gpu.yml ps
curl http://127.0.0.1:8010/health
Если сборка остановилась после загрузки образов
Возможные следующие этапы:
apt-get update
pip install
Это значит, что Docker-образы успешно найдены в локальном кеше, но закрытый компьютер также не имеет доступа к:
Debian-репозиториям;
PyPI;
PaddlePaddle;
PyTorch.
В такой ситуации перенос базовых образов недостаточен. Нужно на компьютере №1 собрать FastService полностью:
CPU:
docker compose -p fastservice build recognition admin
docker pull postgres:16-alpine
GPU:
docker compose \
  -p fastservice \
  -f docker-compose.gpu.yml \
  build recognition admin

docker pull postgres:16-alpine
Затем сохранить готовые образы:
docker save \
  -o fastservice-ready.tar \
  fastservice-recognition:latest \
  fastservice-admin:latest \
  postgres:16-alpine
На компьютере №2:
sudo docker load -i fastservice-ready.tar
Запустить CPU:
sudo docker compose -p fastservice \
  up -d --no-build --pull never
Запустить GPU:
sudo docker compose \
  -p fastservice \
  -f docker-compose.gpu.yml \
  up -d --no-build --pull never
Для вашего первоначального случая сначала переносите базовые образы. Если после этого сборка доходит до apt-get или pip, значит проблема с Docker Hub решена.
