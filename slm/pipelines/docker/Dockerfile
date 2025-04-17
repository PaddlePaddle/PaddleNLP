FROM paddlepaddle/paddlenlp:pipelines-cpu-1.0
COPY start.sh /root/start.sh
COPY create_index.sh /root/PaddleNLP/pipelines
COPY run_server.sh  /root/PaddleNLP/pipelines
COPY run_client.sh /root/PaddleNLP/pipelines
COPY semantic_search.yaml /root/PaddleNLP/pipelines
COPY dense_qa.yaml /root/PaddleNLP/pipelines
RUN chmod +x /root/start.sh
WORKDIR /root
RUN chmod +x /root/start.sh
ENTRYPOINT /root/start.sh && tail -f /dev/null