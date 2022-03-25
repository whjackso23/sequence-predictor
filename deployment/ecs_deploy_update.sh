#!/bin/bash


# Run via Jenkins.
# Used for updates to ECS components of a running Service

# Creates ECS infrastructure for crul-predictor service.
# 1. Target Group
# 2. Task Definition
# 3. Load Balancer Rule

# Updates
# 1. ECS Service



# exit when any command fails
set -e

if [ $# -ne 2 ]; then
  echo "Invalid Arguments, please provide environment and default region as an arguments to continue the ECS deployment."
else
  if [ ${1} == "tst" ]; then
    . ./config/config-tst.ini
    echo "****************************"
    echo "Launching config-tst.ini"
    echo "****************************"
  elif [ ${1} == "dev" ]; then
    . ./config/config-dev.ini
    echo "****************************"
    echo "Launching config-dev.ini"
    echo "****************************"
  elif [ ${1} == "prd" ]; then
    . ./config/config-prd.ini
    echo "****************************"
    echo "Launching config-prd.ini"
    echo "****************************"
  else
    echo "${1} ,Invalid environment argument provided"
  fi

  default_region=${2}

  sed -i -e "s/CLUSTER_NAME/${cluster_name}/" -e "s/CAPACITY_PROVIDER/${capacity_provider}/" update_ecs_service.json
  cat update_ecs_service.json
  aws ecs update-service --cli-input-json file://update_ecs_service.json --force-new-deployment
  echo "update service completed"

fi
