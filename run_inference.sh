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
   echo -e "\t-s steps: 2"
   echo -e "\t-c checkpoint: gs://maxtext-bkt/prod/checkpoint/0/default/d"
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
tpu_name="${USER}-tpu-2"

# Default is converted model - commented code shows default checkpoint for
checkpoint="gs://${USER}-maxtext-outputs/prod-convert-${model}-model-int8-cast-false/checkpoints/0/default"

while getopts "nrbdm:i:s:c:t:" opt
do
   case "$opt" in
      n ) dry_run=true ;;
      r ) runner=true ;;
      b ) bash_setup=true ;;
      m ) model="$OPTARG" ;;
      i ) runid="$OPTARG" ;;
      s ) steps="$OPTARG" ;;
      c ) checkpoint="$OPTARG" ;;
      t ) tpu_name="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


# Begin script in case all parameters are correct
echo
echo "Running: ./run_inference.sh dry_run=${dry_run} runner=${runner} model=${model} runid=${runid} steps=${steps} checkpoint=${checkpoint} bash_setup=${bash_setup}"
echo

dataset_path=gs://${USER}-maxtext-datasets
output_path=gs://${USER}-maxtext-outputs
config="MaxText/configs/benchmark/${model}_base.yml"
run_script="MaxText/decode.py"
run_name=${runid}-decode-${model}-model-int8
options="run_name=${run_name} ici_tensor_parallelism=8 metrics_file='metrics.txt' steps=${steps} load_parameters_path=${checkpoint}  base_output_directory=${output_path} dataset_path=${dataset_path}"
xla_flags="LIBTPU_INIT_ARGS='--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE'"
setup=""
if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi
if "$bash_setup"; then
    setup='bash setup.sh;'
fi

if "$runner"; then
  command="${setup} export ${xla_flags}; python3 ${run_script} ${config} ${options}"
  echo python3 multihost_runner.py --TPU_PREFIX="${tpu_name}" --COMMAND=\"${command}\" --ZONE=us-central1-a
  echo
  $cmd python3 multihost_runner.py --TPU_PREFIX="${tpu_name}" --COMMAND="${command}" --ZONE=us-central1-a
else
  echo export ${xla_flags}
  echo
  $cmd export ${xla_flags}

  echo $setup
  echo
  $cmd $setup

  echo python3 ${run_script} ${config} ${options}
  echo
  $cmd python3 ${run_script} ${config} ${options}
fi
