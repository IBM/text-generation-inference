## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=9.3-1361.1699548029
ARG PROTOC_VERSION=25.0
ARG PYTORCH_INDEX="https://download.pytorch.org/whl"
#ARG PYTORCH_INDEX="https://download.pytorch.org/whl/nightly"
ARG PYTORCH_VERSION=2.1.0

## Base Layer ##################################################################
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} as base
WORKDIR /app

RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
        # we install newer version of requests via pip
        python3.11-requests \
    && dnf install -y make \
        # to help with debugging
        procps \
    && dnf clean all

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

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y \
        cuda-cudart-11-8-${NV_CUDA_CUDART_VERSION} \
        cuda-compat-11-8-${NV_CUDA_COMPAT_VERSION} \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && dnf clean all

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

## CUDA Runtime ################################################################
FROM cuda-base as cuda-runtime

ENV NV_NVTX_VERSION=11.8.86-1 \
    NV_LIBNPP_VERSION=11.8.0.86-1 \
    NV_LIBCUBLAS_VERSION=11.11.3.6-1 \
    NV_LIBNCCL_PACKAGE_VERSION=2.15.5-1+cuda11.8

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y \
        cuda-libraries-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-nvtx-11-8-${NV_NVTX_VERSION} \
        libnpp-11-8-${NV_LIBNPP_VERSION} \
        libcublas-11-8-${NV_LIBCUBLAS_VERSION} \
        libnccl-${NV_LIBNCCL_PACKAGE_VERSION} \
    && dnf clean all

## CUDA Development ############################################################
FROM cuda-base as cuda-devel

ENV NV_CUDA_CUDART_DEV_VERSION=11.8.89-1 \
    NV_NVML_DEV_VERSION=11.8.86-1 \
    NV_LIBCUBLAS_DEV_VERSION=11.11.3.6-1 \
    NV_LIBNPP_DEV_VERSION=11.8.0.86-1 \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.15.5-1+cuda11.8

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf install -y \
        cuda-command-line-tools-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-11-8-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-11-8-${NV_CUDA_CUDART_DEV_VERSION} \
        cuda-nvml-devel-11-8-${NV_NVML_DEV_VERSION} \
        libcublas-devel-11-8-${NV_LIBCUBLAS_DEV_VERSION} \
        libnpp-devel-11-8-${NV_LIBNPP_DEV_VERSION} \
        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all

ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"

## Rust builder ################################################################
# Specific debian version so that compatible glibc version is used
FROM rust:1.73-bullseye as rust-builder
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

RUN dnf install -y make unzip python3.11 python3.11-pip gcc openssl-devel gcc-c++ && \
    dnf clean all && \
    ln -fs /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/local/bin/python && ln -s /usr/bin/pip3.11 /usr/local/bin/pip

RUN pip install --upgrade pip && pip install pytest && pip install pytest-asyncio

# CPU only
ENV CUDA_VISIBLE_DEVICES=""

## Tests #######################################################################
FROM test-base as cpu-tests
ARG PYTORCH_INDEX
ARG PYTORCH_VERSION
ARG SITE_PACKAGES=/usr/local/lib/python3.11/site-packages

WORKDIR /usr/src

# Install specific version of torch
RUN pip install torch=="$PYTORCH_VERSION+cpu" --index-url "${PYTORCH_INDEX}/cpu" --no-cache-dir

COPY server/Makefile server/Makefile

# Install server
COPY proto proto
COPY server server
RUN cd server && \
    make gen-server && \
    pip install ".[accelerate]" --no-cache-dir

# Patch codegen model changes into transformers 4.34
RUN cp server/transformers_patch/modeling_codegen.py ${SITE_PACKAGES}/transformers/models/codegen/modeling_codegen.py

# Install router
COPY --from=router-builder /usr/local/cargo/bin/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=launcher-builder /usr/local/cargo/bin/text-generation-launcher /usr/local/bin/text-generation-launcher

# Install integration tests
COPY integration_tests integration_tests
RUN cd integration_tests && make install

## Python builder #############################################################
FROM cuda-devel as python-builder
ARG PYTORCH_INDEX
ARG PYTORCH_VERSION

RUN dnf install -y unzip git ninja-build && dnf clean all

RUN cd ~ && \
    curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-py311_23.9.0-0-Linux-x86_64.sh && \
    chmod +x Miniconda3-*-Linux-x86_64.sh && \
    bash ./Miniconda3-*-Linux-x86_64.sh -bf -p /opt/miniconda && \
    /opt/miniconda/bin/conda update -y --all && \
    /opt/miniconda/bin/conda update -y cryptography && \
    /opt/miniconda/bin/conda clean -y --all

# Remove tests directory containing test private keys
# conda clean will clean this directory but just in case, it will check the directory existence and remove it
RUN if [ -d " /opt/miniconda/pkgs/conda-content-trust-*/info/test/tests" ]; then rm -rf "/opt/miniconda/pkgs/conda-content-trust-*/info/test/tests"; fi

ENV PATH=/opt/miniconda/bin:$PATH

# Install specific version of torch
RUN pip install ninja==1.11.1.1 --no-cache-dir
RUN pip install torch==$PYTORCH_VERSION+cu118 --index-url "${PYTORCH_INDEX}/cu118" --no-cache-dir


## Build flash attention v2 ####################################################
FROM python-builder as flash-att-v2-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att-v2 Makefile
RUN MAX_JOBS=4 make build-flash-attention-v2

## Build flash attention  ######################################################
FROM python-builder as flash-att-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att Makefile
RUN make build-flash-attention


## Build libraries #############################################################
FROM python-builder as build

# Build custom kernels
COPY server/custom_kernels/ /usr/src/.
RUN cd /usr/src && python setup.py build_ext && python setup.py install


## Build transformers exllama kernels ##########################################
FROM python-builder as exllama-kernels-builder

WORKDIR /usr/src

COPY server/exllama_kernels/ .
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

## Build transformers exllamav2 kernels ########################################
FROM python-builder as exllamav2-kernels-builder

WORKDIR /usr/src

COPY server/exllamav2_kernels/ .
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" python setup.py build

## Flash attention cached build image ##########################################
FROM base as flash-att-cache
COPY --from=flash-att-builder /usr/src/flash-attention/build /usr/src/flash-attention/build
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/layer_norm/build /usr/src/flash-attention/csrc/layer_norm/build
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/rotary/build /usr/src/flash-attention/csrc/rotary/build


## Flash attention v2 cached build image #######################################
FROM base as flash-att-v2-cache
COPY --from=flash-att-v2-builder /usr/src/flash-attention-v2/build /usr/src/flash-attention-v2/build


## Final Inference Server image ################################################
FROM cuda-runtime as server-release
ARG SITE_PACKAGES=/opt/miniconda/lib/python3.11/site-packages

RUN dnf update -y

# Install C++ compiler (required at runtime when PT2_COMPILE is enabled)
RUN dnf install -y gcc-c++ && dnf clean all \
    && useradd -u 2000 tgis -m -g 0

SHELL ["/bin/bash", "-c"]

COPY --from=build /opt/miniconda/ /opt/miniconda/

ENV PATH=/opt/miniconda/bin:$PATH

# These could instead come from explicitly cached images

# Copy build artifacts from flash attention builder
COPY --from=flash-att-cache /usr/src/flash-attention/build/lib.linux-x86_64-cpython-* ${SITE_PACKAGES}
COPY --from=flash-att-cache /usr/src/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-cpython-* ${SITE_PACKAGES}
COPY --from=flash-att-cache /usr/src/flash-attention/csrc/rotary/build/lib.linux-x86_64-cpython-* ${SITE_PACKAGES}

# Copy build artifacts from flash attention v2 builder
COPY --from=flash-att-v2-cache /usr/src/flash-attention-v2/build/lib.linux-x86_64-cpython-* ${SITE_PACKAGES}

# Copy build artifacts from exllama kernels builder
COPY --from=exllama-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-* ${SITE_PACKAGES}

# Copy build artifacts from exllamav2 kernels builder
COPY --from=exllamav2-kernels-builder /usr/src/build/lib.linux-x86_64-cpython-* ${SITE_PACKAGES}

# Install server
COPY proto proto
COPY server server
RUN cd server && make gen-server && pip install ".[accelerate, onnx-gpu, quantize]" --no-cache-dir

# Patch codegen model changes into transformers 4.34.0
RUN cp server/transformers_patch/modeling_codegen.py ${SITE_PACKAGES}/transformers/models/codegen/modeling_codegen.py

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
RUN chmod -R g+w ${SITE_PACKAGES}/text_generation_server /usr/src /usr/local/bin

# Run as non-root user by default
USER tgis

EXPOSE ${PORT}
EXPOSE ${GRPC_PORT}

CMD text-generation-launcher
