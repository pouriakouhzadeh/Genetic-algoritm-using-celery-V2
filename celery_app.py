from celery import Celery
# import tasks

CELERYD_PREFETCH_MULTIPLIER = 1
CELERYD_MAX_TASKS_PER_CHILD = 100

app = Celery('currency_trading_tasks',
             broker='amqp://pouria:P1755063881k@192.168.12.10',  # این آدرس برای RabbitMQ محلی است. در صورت نیاز تغییر دهید.
             backend='rpc://',
             include=['tasks']) 

# تنظیمات تایم‌اوت
app.conf.task_time_limit = 86400  # 24 ساعت به ثانیه
app.conf.task_soft_time_limit = 84600  # 23 ساعت و 30 دقیقه به ثانیه