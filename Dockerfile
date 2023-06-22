## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=8.8-854
ARG PROTOC_VERSION=23.2

## Base Layer ##################################################################
FROM registry.access.redhat.com/ubi8/ubi:${BASE_UBI_IMAGE_TAG} as base
WORKDIR /app

RUN dnf install -y --disableplugin=subscription-manager \
    make \
    # to help with debugging
    procps \
    && dnf clean all --disableplugin=subscription-manager

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

## CUDA Base ###################################################################
FROM base as cuda-base

ENV CUDA_VERSION=11.7.1 \
    NV_CUDA_LIB_VERSION=11.7.1-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=11.7.99-1 \
    NV_CUDA_COMPAT_VERSION=515.86.01-1 \
    NV_NVPROF_VERSION=11.7.101-1

RUN dnf config-manager --disableplugin=subscription-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --disableplugin=subscription-manager \
        cuda-cudart-11-7-${NV_CUDA_CUDART_VERSION} \
        cuda-compat-11-7-${NV_CUDA_COMPAT_VERSION} \
        cuda-nvprof-11-7-${NV_NVPROF_VERSION} \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && dnf clean all --disableplugin=subscription-manager

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

## CUDA Runtime ################################################################
FROM cuda-base as cuda-runtime

ENV NV_NVTX_VERSION=11.7.91-1 \
    NV_LIBNPP_VERSION=11.7.4.75-1 \
    NV_LIBCUBLAS_VERSION=11.10.3.66-1 \
    NV_LIBNCCL_PACKAGE_VERSION=2.13.4-1+cuda11.7

RUN dnf config-manager --disableplugin=subscription-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --disableplugin=subscription-manager \
        cuda-libraries-11-7-${NV_CUDA_LIB_VERSION} \
        cuda-nvtx-11-7-${NV_NVTX_VERSION} \
        libnpp-11-7-${NV_LIBNPP_VERSION} \
        libcublas-11-7-${NV_LIBCUBLAS_VERSION} \
        libnccl-${NV_LIBNCCL_PACKAGE_VERSION} \
    && dnf clean all --disableplugin=subscription-manager

## CUDA Development ############################################################
FROM cuda-base as cuda-devel

ENV NV_CUDA_CUDART_DEV_VERSION=11.7.99-1 \
    NV_NVML_DEV_VERSION=11.7.91-1 \
    NV_LIBCUBLAS_DEV_VERSION=11.10.3.66-1 \
    NV_LIBNPP_DEV_VERSION=11.7.4.75-1 \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.13.4-1+cuda11.7

RUN dnf config-manager --disableplugin=subscription-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --disableplugin=subscription-manager make \
        cuda-command-line-tools-11-7-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-11-7-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-11-7-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-11-7-${NV_CUDA_CUDART_DEV_VERSION} \
        cuda-nvml-devel-11-7-${NV_NVML_DEV_VERSION} \
        libcublas-devel-11-7-${NV_LIBCUBLAS_DEV_VERSION} \
        libnpp-devel-11-7-${NV_LIBNPP_DEV_VERSION} \
        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all --disableplugin=subscription-manager

ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"

## Rust builder ################################################################
FROM rust:1.69 as rust-builder
ARG PROTOC_VERSION

# Install protoc, no longer included in prost crate
RUN cd /tmp && \
    curl -L -O https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip && \
    unzip protoc-*.zip -d /usr/local && rm protoc-*.zip

WORKDIR /usr/src

COPY rust-toolchain.toml rust-toolchain.toml

RUN rustup component add rustfmt

## Internal router builder #####################################################
FROM rust-builder as router-builder

COPY proto proto
COPY router router

WORKDIR /usr/src/router

#RUN --mount=type=cache,target=/root/.cargo --mount=type=cache,target=/usr/src/router/target cargo install --path .
RUN cargo install --path .

## Launcher builder ############################################################
FROM rust-builder as launcher-builder

COPY launcher launcher

WORKDIR /usr/src/launcher

#RUN --mount=type=cache,target=/root/.cargo --mount=type=cache,target=/usr/src/launcher/target cargo install --path .
RUN cargo install --path .

## Tests base ##################################################################
FROM base as test-base

RUN dnf install -y --disableplugin=subscription-manager make unzip python39 gcc openssl-devel gcc-c++ python39-devel && \
    dnf clean all --disableplugin=subscription-manager && \
    ln -s /usr/bin/python3 /usr/local/bin/python && ln -s /usr/bin/pip3 /usr/local/bin/pip

RUN pip install --upgrade pip && pip install pytest && pip install pytest-asyncio

# CPU only
ENV CUDA_VISIBLE_DEVICES=""

## Tests #######################################################################
FROM test-base as cpu-tests

WORKDIR /usr/src

# Install specific version of torch
#RUN cd server && make TORCH_VERSION="1.12.1" TORCH_URL="https://download.pytorch.org/whl/cpu" install-torch
RUN pip install torch=="2.0.0" --extra-index-url "https://download.pytorch.org/whl/cpu" --no-cache-dir

# Install specific version of transformers
COPY server/Makefile-transformers server/Makefile
RUN cd server && make install-custom-transformers

# Install optimum - not used in tests for now
COPY server/Makefile-optimum server/Makefile
#RUN cd server && make install-optimum

COPY server/Makefile server/Makefile

# Install server
COPY proto proto
COPY server server
RUN cd server && \
    make gen-server && \
    pip install ".[bnb]" --no-cache-dir

# Install router
COPY --from=router-builder /usr/local/cargo/bin/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=launcher-builder /usr/local/cargo/bin/text-generation-launcher /usr/local/bin/text-generation-launcher

# Install integration tests
COPY integration_tests integration_tests
RUN cd integration_tests && make install

FROM cuda-devel as build

RUN dnf install -y --disableplugin=subscription-manager \
    unzip \
    curl \
    git \
    && dnf clean all --disableplugin=subscription-manager

RUN cd ~ && \
    curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh && \
    chmod +x Miniconda3-*-Linux-x86_64.sh && \
    bash ./Miniconda3-*-Linux-x86_64.sh -bf -p /opt/miniconda

ENV PATH=/opt/miniconda/bin:$PATH

# Install specific version of torch
RUN pip install ninja==1.11.1 torch=="2.0.0+cu117" --extra-index-url "https://download.pytorch.org/whl/cu117" --no-cache-dir

# Install specific version of flash attention
COPY server/Makefile-flash-att server/Makefile
RUN cd server && make install-flash-attention

# Install specific version of transformers
COPY server/Makefile-transformers server/Makefile
RUN cd server && BUILD_EXTENSIONS="True" make install-custom-transformers

# Install optimum
COPY server/Makefile-optimum server/Makefile
RUN cd server && make install-optimum

# Install onnx
COPY server/Makefile-onnx server/Makefile
RUN cd server && make install-onnx

# Install onnx runtime
COPY server/Makefile-onnx-runtime server/Makefile
RUN cd server && make install-onnx-runtime-nightly

COPY server/Makefile server/Makefile

# Install specific version of deepspeed - excluding for now since we are no longer using it and it doesn't work properly
# RUN cd server && make install-deepspeed

## Final Inference Server image ################################################
FROM cuda-runtime as server-release

# These intended to be overridden
ENV MODEL_NAME=bigscience/bloom \
    NUM_GPUS=8 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    DEPLOYMENT_FRAMEWORK=hf_accelerate \
    DTYPE_STR=bfloat16

SHELL ["/bin/bash", "-c"]

COPY --from=build /opt/miniconda/ /opt/miniconda/

ENV PATH=/opt/miniconda/bin:$PATH

# Install server
COPY proto proto
COPY server server
RUN cd server && \
    make gen-server && \
    pip install ".[bnb]" --no-cache-dir

# Install router
COPY --from=router-builder /usr/local/cargo/bin/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=launcher-builder /usr/local/cargo/bin/text-generation-launcher /usr/local/bin/text-generation-launcher

ENV PORT=3000 \
    GRPC_PORT=8033 \
    HOME=/homedir \
    TRANSFORMERS_CACHE="/tmp/transformers_cache"

# Runs as arbitrary user in OpenShift
RUN mkdir /homedir && chmod g+wx /homedir

# Temporary for dev
RUN chmod -R g+w /opt/miniconda/lib/python3.*/site-packages/text_generation_server /usr/src /usr/local/bin \
    /opt/miniconda/lib/python3.*/site-packages/transformers-* /opt/miniconda/lib/python3.*/site-packages/optimum \
    /opt/miniconda/lib/python3.*/site-packages/onnxruntime/transformers

EXPOSE ${PORT}
EXPOSE ${GRPC_PORT}

CMD HF_HUB_OFFLINE=1 HUGGINGFACE_HUB_CACHE=$TRANSFORMERS_CACHE text-generation-launcher --num-shard $NUM_GPUS
