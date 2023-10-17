#!/bin/bash

set -ue #x for debugging

helpFunction()
{
   echo ""
   echo "Usage: $0 "
   echo -e "\t-n dry_run is true "
   echo -e "\t-r runner command"
   echo -e "\t-m model name: test|1b|32b"
   echo -e "\t-i runid: test|dev|prod"
   echo -e "\t-s steps: 3"
   echo -e "\t-p save_period: 2"
   echo -e "\t-b bash_setup"
   echo -e "\t-t tpu_name: ${USER}-tpu-1"
   exit 1 # Exit script after printing help
}

# Default option values
dry_run=false
runner=false
bash_setup=false
model=test
runid=test
steps=5
save_period=3
tpu_name="${USER}-tpu-1"

while getopts "nrbm:i:s:p:t:" opt
do
   case "$opt" in
      n ) dry_run=true ;;
      r ) runner=true ;;
      b ) bash_setup=true ;;
      m ) model="$OPTARG" ;;
      i ) runid="$OPTARG" ;;
      s ) steps="$OPTARG" ;;
      p ) save_period="$OPTARG" ;;
      t ) tpu_name="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


# Begin script in case all parameters are correct
echo
echo "Running: ./run_training.sh dry_run=${dry_run} runner=${runner} model=${model} runid=${runid} steps=${steps} save_period=${save_period}"
echo


dataset_path=gs://${USER}-maxtext-datasets
output_path=gs://${USER}-maxtext-outputs
config="MaxText/configs/benchmark/${model}_base.yml"
run_script="MaxText/train.py"
run_name=${runid}-train-${model}-model-int8-steps-${steps}
options="run_name=${run_name} steps=${steps} save_period=${save_period}  base_output_directory=${output_path} dataset_path=${dataset_path}"
xla_flags="LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE'"
setup=""
if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi
if "$bash_setup"; then
    setup="bash setup;"
fi

if "$runner"; then
  command="${setup} export ${xla_flags}; python3 ${run_script} ${config} ${options}"
  echo python3 multihost_runner.py --TPU_PREFIX="${tpu_name}" --COMMAND=\"${command}\" --ZONE=us-central2-b
  echo
  $cmd python3 multihost_runner.py --TPU_PREFIX="${tpu_name}" --COMMAND="${command}" --ZONE=us-central2-b
else
  echo export ${setup}
  echo
  $cmd export ${setup}

  echo export ${xla_flags}
  echo
  $cmd export ${xla_flags}

  echo python3 ${run_script} ${config} ${options}
  echo
  $cmd python3 ${run_script} ${config} ${options}
fi



