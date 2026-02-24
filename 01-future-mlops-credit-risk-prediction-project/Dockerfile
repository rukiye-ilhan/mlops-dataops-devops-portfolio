# 1. Python 3.11.9 yüklü hafif bir Linux sürümü kullan
FROM python:3.11.9-slim

# 2. create file that name is app in the  container and enter inside it
WORKDIR /app

# 3. Firstly copy of the requriments file for advantage of cache
COPY requirements.txt .

# 4.İnstall libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5.last codes like (app.py, model file etcdocker.) copy them
COPY . .

# 6.open the 80 port out of the world
EXPOSE 80

# 7. start app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]