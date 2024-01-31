#!/bin/bash

# get PAI_TASK_ROLE_LIST (task roles are separated with commas.)
task_roles=$(echo "$PAI_TASK_ROLE_LIST" | tr ',' ' ')

# for loop with task_role
for taskRole in $task_roles; do
    # get task_count
    task_count_var="PAI_TASK_ROLE_TASK_COUNT_${taskRole}"
    eval task_count="\$$task_count_var"

    # for loop with task_index, get environment variables
    for taskIndex in $(seq 0 $((task_count - 1))); do
        host_ip_var="PAI_HOST_IP_${taskRole}_${taskIndex}"
        eval host_ip="\$$host_ip_var"
	echo "$host_ip slots=8"
    done
done
