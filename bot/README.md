## Deploying Chatbot

**Setup**

* Provision g6.2xlarge instance + Ubuntu 20.04 Deep Learning Pytorch AMI + 250GB storage volume
* Unsure HTTP/HTTPS traffic from anywhere enabled
* SSH into EC2

```
# copy bot directory to ec2
$ sudo apt install supervisor nginx -y
$ sudo systemctl enable supervisor
$ sudo systemctl start supervisor
$ cd bot
$ touch .env # paste keys
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ huggingface-cli login # paste key
```

**Test Local Deploy**

```
$ uvicorn main:app
```

**Configure Gunicorn**

```
$ mkdir run
$ mkdir logs
$ touch gunicorn_start
```

Paste config into `gunicorn_start`:

```
#!/bin/bash

NAME=chatbot
DIR=/home/ubuntu/bot
USER=ubuntu
GROUP=ubuntu
WORKERS=1
WORKER_CLASS=uvicorn.workers.UvicornWorker
VENV=$DIR/.venv/bin/activate
BIND=unix:$DIR/run/gunicorn.sock
LOG_LEVEL=error

cd $DIR
source $VENV

exec gunicorn main:app \
  --name $NAME \
  --workers $WORKERS \
  --worker-class $WORKER_CLASS \
  --user=$USER \
  --group=$GROUP \
  --bind=$BIND \
  --log-level=$LOG_LEVEL \
  --log-file=-
```

Give run permissions:

```
$ chmod u+x gunicorn_start
```

**Configure Supervisor**

```
$ sudo nano /etc/supervisor/conf.d/chatbot.conf
```

Paste into nano:

```
[program:chatbot]
command=/home/ubuntu/bot/gunicorn_start
user=ubuntu       
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/home/ubuntu/bot/logs/gunicorn-error.log
```

Validate setup:

```
$ sudo supervisorctl reread # chatbot: available
$ sudo supervisorctl update # chatbot: added process group
$ sudo supervisorctl status chatbot # STARTING

# Wait 1-2 mins
$ sudo supervisorctl status chatbot # RUNNING   pid 41838, uptime 0:00:08
```

**Configure NGINX**

```
$ sudo nano /etc/nginx/sites-available/chatbot
$ sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
$ sudo nginx -t
# nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
# nginx: configuration file /etc/nginx/nginx.conf test is successful
$ sudo systemctl restart nginx
```

**Permissions**

```
$ sudo usermod -aG ubuntu www-data
```

**Code changes**

After code changes you need to restart supervisor

```
sudo supervisorctl restart chatbot
```
