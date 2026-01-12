import os
import multiprocessing

bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
workers = 1
worker_class = "sync"
timeout = 300
keepalive = 5
graceful_timeout = 120
max_requests = 1000
max_requests_jitter = 50

errorlog = "-"
loglevel = "info"
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
