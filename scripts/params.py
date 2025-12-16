import os
from pathlib import Path
from run_cmd import run_cmd

#
# Global variables
#

# Environment parameters
HOME = str(Path.home())
WSK_CLI = "wsk -i"
N_INVOKER = int(run_cmd('cat ../ansible/environments/distributed/hosts | grep "invoker" | grep -v "\[invokers\]" | wc -l'))
REDIS_HOST = run_cmd('cat ../ansible/environments/distributed/hosts | grep -A 1 "\[edge\]" | grep "ansible_host" | awk {}'.format("{'print $1'}"))
REDIS_PORT = 6379
REDIS_PASSWORD = "openwhisk"
COUCH_PROTOCOL = "http"
COUCH_USER = "whisk_admin"
COUCH_PASSWORD = "some_passw0rd"
COUCH_HOST = run_cmd('cat ../ansible/environments/distributed/hosts | grep -A 1 "\[db\]" | grep "ansible_host" | awk {}'.format("{'print $1'}"))
COUCH_PORT = "5984"
COUCH_LINK = "{}://{}:{}@{}:{}/".format(COUCH_PROTOCOL, COUCH_USER, COUCH_PASSWORD, COUCH_HOST, COUCH_PORT)
COOL_DOWN = "refresh_all"
ENV_INTERVAL_LIMIT = 1
MONITOR_INTERVAL=1000
UPDATE_RETRY_TIME = 10
KEEP_ALIVE_TIME = 600 # second
LOG_PATH = os.path.dirname(os.getcwd())+'/demo/logs/'

