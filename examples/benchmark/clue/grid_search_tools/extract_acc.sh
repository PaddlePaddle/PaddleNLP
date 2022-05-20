MODEL_PATH=$1


afqmc=`cat ${MODEL_PATH}/afqmc/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
tnews=`cat ${MODEL_PATH}/tnews/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
ifly=`cat ${MODEL_PATH}/ifly/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
cmnli=`cat ${MODEL_PATH}/cmnli/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
ocnli=`cat ${MODEL_PATH}/ocnli/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
wsc=`cat ${MODEL_PATH}/wsc/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
csl=`cat ${MODEL_PATH}/csl/*|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`

cmrc2018=`cat  mrc/${MODLE_PATH}/cmrc2018_log/workerlog.0|grep best_res|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
chid=`cat  mrc/${MODLE_PATH}/chid_log/workerlog.0|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`
c3=`cat  mrc/${MODLE_PATH}/c3_log/workerlog.0|grep best_acc|awk '{print $2}'|awk '$0>x{x=$0};END{print x}'`

echo  ${afqmc}"\t"${tnews}"\t"${ifly}"\t"${cmnli}"\t"${ocnli}"\t"${wsc}"\t"${csl}"\t"${cmrc2018}"\t"${chid}"\t"${c3}

