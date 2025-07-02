# 0) Сбрасываем любые прокси-аргументы ещё до FROM
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""

FROM python:3.11-bookworm

# 1) Гарантируем внутри контейнера отсутствие прокси
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV NO_PROXY="host.docker.internal,127.0.0.1"

# 2) Обновляем pip и сразу ставим python-dotenv (пример вашего проекта)
RUN pip install --upgrade pip \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org && \
    pip install python-dotenv \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org

# 3) Системные утилиты и сертификаты
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
      curl \
      ca-certificates \
      libffi-dev \
      libssl-dev \
 && update-ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4) Копируем requirements и ставим зависимости с доверенными хостами
COPY requirements.txt ./
RUN pip install --upgrade pip \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org && \
    pip install \
        --cache-dir=/root/.cache/pip \
        -r requirements.txt \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org
# 5) Копируем всё остальное
COPY . .

# 6) Экспонируем порт и запускаем
EXPOSE 5005
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5005"]
