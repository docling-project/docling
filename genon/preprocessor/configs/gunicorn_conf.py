bind = "0.0.0.0:8080"

workers = 5
worker_class = "uvicorn.workers.UvicornWorker"

BASE_DIR = "/app"
pythonpath = BASE_DIR + "/src"
chdir = BASE_DIR

graceful_timeout = 3600
timeout = 0

loglevel = "info"
accesslog = "-"
errorlog = "-"