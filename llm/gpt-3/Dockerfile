ARG FROM_IMAGE_NAME=nvcr.io/nvidia/paddlepaddle:23.11-py3

FROM ${FROM_IMAGE_NAME} AS build_stage

WORKDIR /opt

# Install TE
ARG TE_COMMIT
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
	cd TransformerEngine && \
    if [ ! -z $TE_COMMIT ]; then \
        git fetch origin $TE_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    git submodule update --init --recursive && \
    pip wheel -v --no-deps ./

# Install PaddleNLP
ADD . /opt/paddlenlp
RUN cd /opt/paddlenlp && \
    pip wheel -v --no-deps .

FROM ${FROM_IMAGE_NAME}

RUN --mount=type=bind,from=build_stage,source=/opt/TransformerEngine,target=/opt/TransformerEngine \
    pip install /opt/TransformerEngine/*.whl

RUN pip install regex tool_helpers visualdl==2.5.3

RUN --mount=type=bind,from=build_stage,source=/opt/paddlenlp,target=/opt/paddlenlp \
    pip install /opt/paddlenlp/*.whl

RUN pip install wandb

WORKDIR /workspace
RUN chmod -R a+w /workspace

ADD ./llm/gpt-3 /workspace/gpt-3