import os

bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
workers = 1
worker_class = "sync"
threads = 1

timeout = 300
keepalive = 5
graceful_timeout = 120

max_requests = 100
max_requests_jitter = 10

worker_tmp_dir = "/dev/shm"

preload_app = False

errorlog = "-"
loglevel = "warning"
accesslog = "-"

def post_fork(server, worker):
    import gc
    gc.collect()

def worker_exit(server, worker):
    import gc
    gc.collect()
