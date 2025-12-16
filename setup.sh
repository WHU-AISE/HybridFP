#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Build and Ansible deploy
echo "Starting Build and Deployment Process..."

cd ansible
ENVIRONMENT=local
sudo ansible-playbook -i environments/local setup.yml

cd ..
sudo ./gradlew distDocker

cd ansible
sudo ansible-playbook -i environments/local couchdb.yml
sudo ansible-playbook -i environments/local initdb.yml
sudo ansible-playbook -i environments/local wipe.yml
sudo ansible-playbook -i environments/local apigateway.yml
sudo ansible-playbook -i environments/local openwhisk.yml
sudo ansible-playbook -i environments/local postdeploy.yml

cd ..
wsk property set --apihost https://172.17.0.1:443
wsk property set --auth "$(cat ./ansible/files/auth.guest)"

cd ../applications
./deploy_functions.sh

