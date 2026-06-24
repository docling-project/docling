bind = "0.0.0.0:8080"

workers = 1
worker_class = "uvicorn.workers.UvicornWorker"

BASE_DIR = "/app"
pythonpath = BASE_DIR + "/src"
chdir = BASE_DIR
