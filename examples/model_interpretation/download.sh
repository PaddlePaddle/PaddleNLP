wget https://paddlenlp.bj.bcebos.com/data/model_interpretation.tar
wait
tar -xvf model_interpretation.tar
wait
mv ./model_interpretation/vocab.char ./task/similarity/simnet/
mv ./model_interpretation/vocab_QQP ./task/similarity/simnet/
mv ./model_interpretation/simnet_vocab.txt ./task/similarity/simnet/

mv ./model_interpretation/vocab.sst2_train ./task/senti/rnn/
mv ./model_interpretation/vocab.txt ./task/senti/rnn