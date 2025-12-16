#! /bin/bash

cd ./ansible
ansible-playbook -i environments/local openwhisk.yml -e mode=clean
