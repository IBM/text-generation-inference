## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=8.8-1009
ARG PROTOC_VERSION=23.4
ARG PYTORCH_VERSION=2.1.0.dev20230730
ARG OPTIMUM_VERSION=1.9.1

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

ENV CUDA_VERSION=11.8.0 \
    NV_CUDA_LIB_VERSION=11.8.0-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=11.8.89-1 \
    NV_CUDA_COMPAT_VERSION=520.61.05-1

RUN dnf config-manager --disableplugin=subscription-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --disableplugin=subscription-manager \
        cuda-cudart-11-8-${NV_CUDA_CUDART_VERSION} \
        cuda-compat-11-8-${NV_CUDA_COMPAT_VERSION} \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && dnf clean all --disableplugin=subscription-manager

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

## CUDA Runtime ################################################################
FROM cuda-base as cuda-runtime

ENV NV_NVTX_VERSION=11.8.86-1 \
    NV_LIBNPP_VERSION=11.8.0.86-1 \
    NV_LIBCUBLAS_VERSION=11.11.3.6-1 \
    NV_LIBNCCL_PACKAGE_VERSION=2.15.5-1+cuda11.8

RUN dnf config-manager --disableplugin=subscription-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --disableplugin=subscription-manager \
        cuda-libraries-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-nvtx-11-8-${NV_NVTX_VERSION} \
        libnpp-11-8-${NV_LIBNPP_VERSION} \
        libcublas-11-8-${NV_LIBCUBLAS_VERSION} \
        libnccl-${NV_LIBNCCL_PACKAGE_VERSION} \
    && dnf clean all --disableplugin=subscription-manager

## CUDA Development ############################################################
FROM cuda-base as cuda-devel

ENV NV_CUDA_CUDART_DEV_VERSION=11.8.89-1 \
    NV_NVML_DEV_VERSION=11.8.86-1 \
    NV_LIBCUBLAS_DEV_VERSION=11.11.3.6-1 \
    NV_LIBNPP_DEV_VERSION=11.8.0.86-1 \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.15.5-1+cuda11.8

RUN dnf config-manager --disableplugin=subscription-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y --disableplugin=subscription-manager make \
        cuda-command-line-tools-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-11-8-${NV_CUDA_CUDART_DEV_VERSION} \
        cuda-nvml-devel-11-8-${NV_NVML_DEV_VERSION} \
        libcublas-devel-11-8-${NV_LIBCUBLAS_DEV_VERSION} \
        libnpp-devel-11-8-${NV_LIBNPP_DEV_VERSION} \
        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all --disableplugin=subscription-manager

ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"

## Rust builder ################################################################
# Specific debian version so that compatible glibc version is used
FROM rust:1.71-buster as rust-builder
ARG PROTOC_VERSION

ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

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
ARG PYTORCH_VERSION
ARG OPTIMUM_VERSION

WORKDIR /usr/src

# Install specific version of torch
RUN pip install torch=="$PYTORCH_VERSION+cpu" --index-url "https://download.pytorch.org/whl/nightly/cpu" --no-cache-dir

# Install specific version of transformers
COPY server/Makefile-transformers server/Makefile
RUN cd server && make install-custom-transformers

# Install optimum - not used in tests for now
#RUN pip install optimum==$OPTIMUM_VERSION --no-cache-dir

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

## Build #######################################################################
FROM cuda-devel as build
ARG PYTORCH_VERSION
ARG OPTIMUM_VERSION

RUN dnf install -y --disableplugin=subscription-manager \
    unzip \
    curl \
    git \
    && dnf clean all --disableplugin=subscription-manager

RUN cd ~ && \
    curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh && \
    chmod +x Miniconda3-*-Linux-x86_64.sh && \
    bash ./Miniconda3-*-Linux-x86_64.sh -bf -p /opt/miniconda

# Remove tests directory containing test private keys
RUN rm -r /opt/miniconda/pkgs/conda-content-trust-*/info/test/tests

ENV PATH=/opt/miniconda/bin:$PATH

# Install specific version of torch
RUN pip install ninja==1.11.1
RUN pip install torch==$PYTORCH_VERSION+cu118 --index-url "https://download.pytorch.org/whl/nightly/cu118" --no-cache-dir

# Install specific version of flash attention
COPY server/Makefile-flash-att server/Makefile
RUN cd server && make install-flash-attention

# Install specific version of transformers
COPY server/Makefile-transformers server/Makefile
RUN cd server && BUILD_EXTENSIONS="True" make install-custom-transformers

# Install optimum
RUN pip install optimum[onnxruntime-gpu]==$OPTIMUM_VERSION --no-cache-dir

# Install onnx
COPY server/Makefile-onnx server/Makefile
RUN cd server && make install-onnx

# Install onnx runtime
COPY server/Makefile-onnx-runtime server/Makefile
RUN cd server && make install-onnx-runtime

COPY server/Makefile server/Makefile

# Install specific version of deepspeed - excluding for now since we are no longer using it and it doesn't work properly
# RUN cd server && make install-deepspeed

## Final Inference Server image ################################################
FROM cuda-runtime as server-release

# Install C++ compiler (required at runtime when PT2_COMPILE is enabled)
RUN dnf install -y --disableplugin=subscription-manager gcc-c++ \
    && dnf clean all --disableplugin=subscription-manager \
    && useradd -u 2000 tgis -m -g 0

SHELL ["/bin/bash", "-c"]

COPY --from=build /opt/miniconda/ /opt/miniconda/

ENV PATH=/opt/miniconda/bin:$PATH

# Install server
COPY proto proto
COPY server server
RUN cd server && \
    make gen-server && \
    pip install ".[bnb, accelerate]" --no-cache-dir

# Install router
COPY --from=router-builder /usr/local/cargo/bin/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=launcher-builder /usr/local/cargo/bin/text-generation-launcher /usr/local/bin/text-generation-launcher

ENV PORT=3000 \
    GRPC_PORT=8033 \
    HOME=/home/tgis

# Runs as arbitrary user in OpenShift
RUN chmod -R g+rwx ${HOME}

# Temporary for dev
RUN chmod -R g+w /opt/miniconda/lib/python3.*/site-packages/text_generation_server /usr/src /usr/local/bin \
    /opt/miniconda/lib/python3.*/site-packages/transformers-* /opt/miniconda/lib/python3.*/site-packages/optimum \
    /opt/miniconda/lib/python3.*/site-packages/onnxruntime/transformers

# Run as non-root user by default
USER tgis

EXPOSE ${PORT}
EXPOSE ${GRPC_PORT}

CMD text-generation-launcher
