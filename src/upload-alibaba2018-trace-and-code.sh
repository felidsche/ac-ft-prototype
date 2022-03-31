#!/bin/bash

# connect to VPN
sudo openvpn \
  --config config.ovpn \
  --redirect-gateway # the last flag is only necessary when down- or uploading

# upload batch-task.tar.gz to HDFS (Note: no filename is needed)
hdfscli upload \
  --alias=prod \
  ../data/alibaba_clusterdata_v2018/batch_task.tar.gz \
  spark/alibaba2018-trace -v

# same for batch-instance.tar.gz
hdfscli upload \
  --alias=prod \
  ../data/alibaba_clusterdata_v2018/batch_instance.tar.gz \
  spark/alibaba2018-trace -v

# upload python for clean (DONE)
hdfscli upload \
  --alias=prod \
  clean/failed_instances.py \
  spark/alibaba2018-trace -f

# upload jar for model (allow overwriting)
hdfscli upload \
  --alias=prod \
  model/scala/target/scala-1.0-SNAPSHOT-jar-with-dependencies.jar \
  jar-files/ -f

# upload TC SA script
hdfscli upload \
  --alias=prod \
  analysis/tc_sensitivity_analysis.py \
  spark/alibaba2018-trace -f
