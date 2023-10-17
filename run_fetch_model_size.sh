#!/bin/bash

set -ue #x for debugging

helpFunction()
{
   echo ""
   echo "Usage: $0 "
   echo -e "\t-n dry_run is true "
   echo -e "\t-m model name: test|1b|32b"
   echo -e "\t-i runid: test|dev|prod"
   exit 1 # Exit script after printing help
}

# Default option values
dry_run=false
model=32b
runid=prod

while getopts "nm:i:t:" opt
do
   case "$opt" in
      n ) dry_run=true ;;
      m ) model="$OPTARG" ;;
      i ) runid="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

echo
echo "Printing trained model size"
gsutil du -sh gs://msingh-maxtext-outputs/${runid}-train-${model}-model-int8-steps-200/checkpoints/150/default/d
echo
echo "Printing converted model size"
gsutil du -sh gs://msingh-maxtext-outputs/${runid}-convert-${model}-model-int8-cast-false/checkpoints/0/default/d
echo
gsutil du -sh gs://msingh-maxtext-outputs/${runid}-convert-${model}-model-int8-cast-true/checkpoints/0/default/d
echo
