#!/bin/bash

# requires VPN connection and kubectl to point to the cluster
sudo openvpn \
  --config config.ovpn

# requires upload-alibaba2018-trace-and-code.sh to be done (DONE 31.12.2021)

# clean the cluster trace (..and wait for it to finish) -> took 2 hours on 29.12.2021, result is 3.5gb
kubectl apply -f clean/trace.yaml

# job gut stuck in success mode -> save logs
kubectl logs pod/clean-alibaba-2018-cluster-trace-sample-46c7797e07208b99-exec-1 >../logs/clean/clean-alibaba-2018-cluster-trace-sample-46c7797e07208b99-exec-1.log
kubectl logs pod/clean-alibaba-2018-cluster-trace-sample-46c7797e07208b99-exec-2 >../logs/clean/clean-alibaba-2018-cluster-trace-sample-46c7797e07208b99-exec-2.log
kubectl logs pod/clean-alibaba-2018-cluster-trace-sample-46c7797e07208b99-exec-3 >../logs/clean/clean-alibaba-2018-cluster-trace-sample-46c7797e07208b99-exec-3.log
kubectl logs pod/clean-alibaba2018-trace-driver >../logs/clean/clean-alibaba2018-trace-driver.log

# delete the spark app

# port forwarding of the spark history server service
kubectl port-forward service/spark-history-server-web 18081:18080 # localhost:cluster

kubectl port-forward service/clean-alibaba2018-trace-ui-svc 4040:4040 # localhost:cluster

# do nested cross validation on the GBT binary classifier (..and wait for it to finish)
kubectl apply -f model/GBTNestedCVTaskFailBinPred.yaml
# NOTE: this only works on a sample

# do cross validation on the tuned GBT binary classifier (..and wait for it to finish)
kubectl apply -f model/GBTCVTaskFailBinPred.yaml

# BEFORE PROCEEDING:
# - TODO get prediction time from Spark job event log (model evaluation duration)
# - TODO enter it in analysis/tc_sensitivity_analysis.py for TP

# download the model
# model selection
hdfscli download \
  --alias=prod \
  spark/alibaba2018-trace/out/GBT5CV3parall_priorFeat_tune_maxDepth_Iter_Bins_03inst777Seed_select \
  ../out/model/dump/final/

# model training
hdfscli download \
  --alias=prod \
  spark/alibaba2018-trace/out/GBT5CV3parall_priorFeat_tune_maxDepth_Iter_Bins_03inst777Seed \
  ../out/model/dump/final/

# download the test set
hdfscli download \
  --alias=prod \
  spark/alibaba2018-trace/out/GBT5CV3parall_priorFeat_tune_maxDepth_Iter_Bins_03inst777SeedVal \
  ../out/model/eval/final/

# it is not possible to plot on the cluster so download the clean data
hdfscli download \
  --alias=prod \
  spark/alibaba2018-trace/out/batch_jobs_clean_01inst_1task_0006S_1F \
  ../out/clean/

hdfscli download \
  --alias=prod \
  spark/alibaba2018-trace/out/batch_jobs_clean_02inst_1task_00015S_1F \
  ../out/clean/

hdfscli download \
  --alias=prod \
  spark/alibaba2018-trace/out/batch_fail \
  ../out/clean/

# do tc sensitivity analysis and plot the results
kubectl apply -f analysis/tc_sensitivity_analysis.yaml
