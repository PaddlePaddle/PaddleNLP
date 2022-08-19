# linux
FROM w5688414/pipelines-cpu-linux:1.3
# FROM w5688414/pipelines-cpu-win:1.3
COPY start.sh /root/start.sh
COPY create_index.sh /root/PaddleNLP/applications/experimental/pipelines
COPY run_server.sh  /root/PaddleNLP/applications/experimental/pipelines 
COPY run_client.sh /root/PaddleNLP/applications/experimental/pipelines
RUN chmod +x /root/start.sh
WORKDIR /root
RUN chmod +x /root/start.sh
ENTRYPOINT /root/start.sh && tail -f /dev/null
