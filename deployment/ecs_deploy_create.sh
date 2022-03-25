#!/bin/bash

# Run via Jenkins.
# Used to create new ECS components and Service

# Creates ECS infrastructure for crul-predictor service.
# 1. Target Group
# 2. Task Definition
# 3. Load Balancer Rule
# 4. ECS Service


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

  sed -i "s/VPC_ID/$vpc_id/g" create_tg.json

  #To create target group
  aws elbv2 create-target-group --cli-input-json file://create_tg.json

  echo " Target group created successfully in ${1} ."

  target_gp_arn=$(aws elbv2 describe-target-groups --target-group-arns --output text --query 'TargetGroups[*].[TargetGroupName,TargetGroupArn]' | grep "crul-predict-service-tg" | awk '{print $2}')

  FORMATTED_TARGET_GROUP_ARN=$(printf '%s\n' "$target_gp_arn" | sed -e 's/[\/&]/\\&/g')

  sed -i "s/TARGET_GROUP_ARN/$FORMATTED_TARGET_GROUP_ARN/g" create_ecs_service.json

  sed -i "s/TARGET_GROUP_ARN/$FORMATTED_TARGET_GROUP_ARN/g" actions_fixed_response.json

  sed -i "s/VPC_ID/$vpc_id/g" create_tg.json

  sed -i -e "s/ACCOUNT/${account}/" -e "s/REGION/${default_region}/" -e "s/ARTIFACT_BUCKET/${artifact_bucket}/" task_definition.json

  #To register task definition
  aws ecs register-task-definition \
    --execution-role-arn ${task_exec_role_arn} \
    --cli-input-json file://task_definition.json

  echo " Task definition registered successfully in ${1} ."

  #To create ELB Listener rule
  aws elbv2 create-rule \
    --listener-arn ${listener_arn} \
    --priority 9 \
    --conditions file://listener_rule.json \
    --actions file://actions_fixed_response.json

  echo " ELB listener rule created successfully in ${1} ."

  sed -i -e "s/CLUSTER_NAME/${cluster_name}/g" -e "s/SERVICE_NAME/${service_name}/g" create_ecs_service.json

  sed -i -e "s/CLUSTER_NAME/${cluster_name}/g" -e "s/SERVICE_NAME/${service_name}/g" update_ecs_service.json

  aws ecs create-service --cli-input-json file://create_ecs_service.json

  echo "Service ${service_name} created in ${cluster_name}"
fi
