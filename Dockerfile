## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=9.3-1610
ARG PROTOC_VERSION=25.2
ARG PYTORCH_INDEX="https://download.pytorch.org/whl"
# ARG PYTORCH_INDEX="https://download.pytorch.org/whl/nightly"
ARG AUTO_GPTQ_VERSION=0.7.1

# match PyTorch version that was used to compile flash-attention v2 pre-built wheels
# e.g. flash-attn v2.5.2 => torch ['1.12.1', '1.13.1', '2.0.1', '2.1.2', '2.2.0', '2.3.0.dev20240126']
# https://github.com/Dao-AILab/flash-attention/blob/v2.5.2/.github/workflows/publish.yml#L47
# use nightly build index for torch .dev pre-release versions
ARG PYTORCH_VERSION=2.2.1

ARG PYTHON_VERSION=3.11


## Base Layer ##################################################################
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG} as base
WORKDIR /app

ARG PYTHON_VERSION

RUN dnf remove -y --disableplugin=subscription-manager \
        subscription-manager \
        # we install newer version of requests via pip
    python${PYTHON_VERSION}-requests \
    && dnf install -y make \
        # to help with debugging
        procps \
    && dnf clean all

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8


## CUDA Base ###################################################################
FROM base as cuda-base

# Ref: https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/
ENV CUDA_VERSION=12.1.0 \
    NV_CUDA_LIB_VERSION=12.1.0-1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NV_CUDA_CUDART_VERSION=12.1.55-1 \
    NV_CUDA_COMPAT_VERSION=530.30.02-1

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-cudart-12-1-${NV_CUDA_CUDART_VERSION} \
        cuda-compat-12-1-${NV_CUDA_COMPAT_VERSION} \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf \
    && dnf clean all

ENV CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/nvidia/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"


## CUDA Development ############################################################
FROM cuda-base as cuda-devel

# Ref: https://developer.nvidia.com/nccl/nccl-legacy-downloads
ENV NV_CUDA_CUDART_DEV_VERSION=12.1.55-1 \
    NV_NVML_DEV_VERSION=12.1.55-1 \
    NV_LIBCUBLAS_DEV_VERSION=12.1.0.26-1 \
    NV_LIBNPP_DEV_VERSION=12.0.2.50-1 \
    NV_LIBNCCL_DEV_PACKAGE_VERSION=2.18.3-1+cuda12.1

RUN dnf config-manager \
       --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo \
    && dnf install -y \
        cuda-command-line-tools-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-libraries-devel-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-minimal-build-12-1-${NV_CUDA_LIB_VERSION} \
        cuda-cudart-devel-12-1-${NV_CUDA_CUDART_DEV_VERSION} \
        cuda-nvml-devel-12-1-${NV_NVML_DEV_VERSION} \
        libcublas-devel-12-1-${NV_LIBCUBLAS_DEV_VERSION} \
        libnpp-devel-12-1-${NV_LIBNPP_DEV_VERSION} \
        libnccl-devel-${NV_LIBNCCL_DEV_PACKAGE_VERSION} \
    && dnf clean all

ENV LIBRARY_PATH="$CUDA_HOME/lib64/stubs"


## Rust builder ################################################################
# Specific debian version so that compatible glibc version is used
FROM rust:1.77-bullseye as rust-builder
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

ARG GIT_COMMIT_HASH
COPY launcher launcher

WORKDIR /usr/src/launcher

#RUN --mount=type=cache,target=/root/.cargo --mount=type=cache,target=/usr/src/launcher/target cargo install --path .
RUN env GIT_COMMIT_HASH=${GIT_COMMIT_HASH} cargo install --path .


## Tests base ##################################################################
FROM base as test-base

ARG PYTHON_VERSION

RUN dnf install -y make unzip python${PYTHON_VERSION} python${PYTHON_VERSION}-pip gcc openssl-devel gcc-c++ git && \
    dnf clean all && \
    ln -fs /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python && ln -s /usr/bin/pip${PYTHON_VERSION} /usr/local/bin/pip

RUN pip install --upgrade pip --no-cache-dir && pip install pytest --no-cache-dir && pip install pytest-asyncio --no-cache-dir

# CPU only
ENV CUDA_VISIBLE_DEVICES=""


## Tests #######################################################################
FROM test-base as cpu-tests
ARG PYTORCH_INDEX
ARG PYTORCH_VERSION
ARG PYTHON_VERSION
ARG SITE_PACKAGES=/usr/local/lib/python${PYTHON_VERSION}/site-packages

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

# Patch codegen model changes into transformers
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
ARG PYTHON_VERSION
ARG MINIFORGE_VERSION=23.11.0-0

# consistent arch support anywhere we compile CUDA code
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX;8.9"

RUN dnf install -y unzip git ninja-build && dnf clean all

RUN curl -fsSL -v -o ~/miniforge3.sh -O  "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-$(uname)-$(uname -m).sh" && \
    chmod +x ~/miniforge3.sh && \
    bash ~/miniforge3.sh -b -p /opt/conda && \
    source "/opt/conda/etc/profile.d/conda.sh" && \
    conda create -y -p /opt/tgis python=${PYTHON_VERSION} && \
    conda activate /opt/tgis && \
    rm ~/miniforge3.sh

ENV PATH=/opt/tgis/bin/:$PATH

# Install specific version of torch
RUN pip install ninja==1.11.1.1 --no-cache-dir
RUN pip install packaging --no-cache-dir
RUN pip install torch==$PYTORCH_VERSION+cu121 --index-url "${PYTORCH_INDEX}/cu121" --no-cache-dir


## Build flash attention v2 ####################################################
FROM python-builder as flash-att-v2-builder
ARG FLASH_ATT_VERSION=v2.5.6

WORKDIR /usr/src/flash-attention-v2

# Download the wheel or build it if a pre-compiled release doesn't exist
# MAX_JOBS: For CI, limit number of parallel compilation threads otherwise the github runner goes OOM
RUN MAX_JOBS=2  pip --verbose wheel --no-deps flash-attn==${FLASH_ATT_VERSION} \
    "git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATT_VERSION}#subdirectory=csrc/layer_norm" \
    "git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATT_VERSION}#subdirectory=csrc/rotary" \
    --no-build-isolation --no-cache-dir


## Install auto-gptq ###########################################################
## Uncomment if a custom autogptq build is required
#FROM python-builder as auto-gptq-installer
#ARG AUTO_GPTQ_REF=896d8204bc89a7cfbda42bf3314e13cf4ce20b02
#
#WORKDIR /usr/src/auto-gptq-wheel
#
## numpy is required to run auto-gptq's setup.py
#RUN pip install numpy
#RUN DISABLE_QIGEN=1 pip wheel git+https://github.com/AutoGPTQ/AutoGPTQ@${AUTO_GPTQ_REF} --no-cache-dir --no-deps --verbose

## Build libraries #############################################################
FROM python-builder as build

# Build custom kernels
COPY server/custom_kernels/ /usr/src/.
RUN cd /usr/src && python setup.py build_ext && python setup.py install


## Flash attention v2 cached build image #######################################
FROM base as flash-att-v2-cache

# Copy just the wheels we built for flash-attention
COPY --from=flash-att-v2-builder /usr/src/flash-attention-v2 /usr/src/flash-attention-v2


## Auto gptq cached build image ################################################
## Uncomment if a custom autogptq build is required
#FROM base as auto-gptq-cache
#
## Copy just the wheel we built for auto-gptq
#COPY --from=auto-gptq-installer /usr/src/auto-gptq-wheel /usr/src/auto-gptq-wheel


## Full set of python installations for server release #########################

FROM python-builder as python-installations

ARG PYTHON_VERSION
ARG AUTO_GPTQ_VERSION
ARG SITE_PACKAGES=/opt/tgis/lib/python${PYTHON_VERSION}/site-packages

COPY --from=build /opt/tgis /opt/tgis

# `pip` is installed in the venv here
ENV PATH=/opt/tgis/bin:$PATH

# Install flash attention v2 from the cache build
RUN --mount=type=bind,from=flash-att-v2-cache,src=/usr/src/flash-attention-v2,target=/usr/src/flash-attention-v2 \
    pip install /usr/src/flash-attention-v2/*.whl --no-cache-dir

# Copy over the auto-gptq wheel and install it
#RUN --mount=type=bind,from=auto-gptq-cache,src=/usr/src/auto-gptq-wheel,target=/usr/src/auto-gptq-wheel \
#    pip install /usr/src/auto-gptq-wheel/*.whl --no-cache-dir

# We only need to install a custom-built auto-gptq version if we need a pre-release
# or are using a PyTorch nightly version
RUN pip install auto-gptq=="${AUTO_GPTQ_VERSION}" --no-cache-dir


# Install ibm-fms from branch (required for spec. decoding)
# TODO: switch this out for fms-extras main once speculative decoding fully in
RUN git clone https://github.com/JRosenkranz/fms-extras.git --branch speculative_decoding_e2e --single-branch && \
    cd fms-extras && git checkout speculative_decoding_e2e && \
    pip install . --verbose --no-cache-dir --no-build-isolation

# Install server
# git is required to pull the fms-extras dependency
RUN dnf install -y git && dnf clean all
COPY proto proto
COPY server server
# Extra url is required to install cuda-12 version of onnxruntime-gpu
# Ref: https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x
RUN cd server && make gen-server && pip install ".[accelerate, ibm-fms, onnx-gpu, quantize]" --no-cache-dir --extra-index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Patch codegen model changes into transformers 4.35
RUN cp server/transformers_patch/modeling_codegen.py ${SITE_PACKAGES}/transformers/models/codegen/modeling_codegen.py


## Final Inference Server image ################################################
FROM base as server-release
ARG PYTHON_VERSION
ARG SITE_PACKAGES=/opt/tgis/lib/python${PYTHON_VERSION}/site-packages

# Install C++ compiler (required at runtime when PT2_COMPILE is enabled)
RUN dnf install -y gcc-c++ && dnf clean all \
    && useradd -u 2000 tgis -m -g 0

# Copy in the full python environment
COPY --from=python-installations /opt/tgis /opt/tgis

ENV PATH=/opt/tgis/bin:$PATH

# Print a list of all installed packages and versions
RUN pip list -v --disable-pip-version-check --no-python-version-warning

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
