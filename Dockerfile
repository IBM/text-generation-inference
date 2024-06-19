## Global Args #################################################################
ARG BASE_UBI_IMAGE_TAG=9.3-1610
ARG PROTOC_VERSION=25.3
ARG PYTORCH_INDEX="https://download.pytorch.org/whl"
# ARG PYTORCH_INDEX="https://download.pytorch.org/whl/nightly"
ARG AUTO_GPTQ_VERSION=0.7.1

ARG PYTORCH_VERSION=2.1.2

ARG PYTHON_VERSION=3.11

# This is overriden in the Makefile such that `-private` is used for CI builds;
# use public by default for local development
ARG CACHE_REGISTRY=docker-na-public.artifactory.swg-devops.com/wcp-ai-foundation-team-docker-virtual
ARG AUTO_GPTQ_CACHE_TAG=auto-gptq-cache.0b32a42

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

## Rust builder ################################################################
# Using bookworm for compilation so the rust binaries get linked against libssl.so.3
FROM rust:1.78-bookworm as rust-builder
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

# temp: install newer transformers lib that optimum clashes with
RUN pip install transformers==4.40.0 tokenizers==0.19.1 --no-cache-dir

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
FROM base as python-builder
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
# There is no rocm 6.0 wheel in the python repo as of March 1st 2023
RUN pip install "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/torch-${PYTORCH_VERSION}+rocm6.1-cp311-cp311-linux_x86_64.whl" "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1/pytorch_triton_rocm-2.1.0+rocm6.1.4d510c3a44-cp311-cp311-linux_x86_64.whl" --no-cache-dir

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
FROM python-builder as build

## Auto gptq cached build image ################################################
## Uncomment if a custom autogptq build is required
#FROM base as auto-gptq-cache
#
## Copy just the wheel we built for auto-gptq
#COPY --from=auto-gptq-installer /usr/src/auto-gptq-wheel /usr/src/auto-gptq-wheel


FROM ${CACHE_REGISTRY}/auto-gptq-cache:${AUTO_GPTQ_CACHE_TAG} as auto-gptq-remote-cache

FROM python-builder as python-installations

ARG PYTHON_VERSION
ARG AUTO_GPTQ_VERSION
ARG SITE_PACKAGES=/opt/tgis/lib/python${PYTHON_VERSION}/site-packages

COPY --from=build /opt/tgis /opt/tgis

# `pip` is installed in the venv here
ENV PATH=/opt/tgis/bin:$PATH

# Copy over the auto-gptq wheel and install it
#RUN --mount=type=bind,from=auto-gptq-cache,src=/usr/src/auto-gptq-wheel,target=/usr/src/auto-gptq-wheel \
#    pip install /usr/src/auto-gptq-wheel/*.whl --no-cache-dir

# We only need to install a custom-built auto-gptq version if we need a pre-release
# or are using a PyTorch nightly version
RUN pip install auto-gptq=="${AUTO_GPTQ_VERSION}" --no-cache-dir

# Install server
# git is required to pull the fms-extras dependency
RUN dnf install -y git && dnf clean all
COPY proto proto
COPY server server
RUN cd server && make gen-server && pip install ".[quantize]" --no-cache-dir

# temp: install newer transformers lib that optimum clashes with
RUN pip install transformers==4.40.0 tokenizers==0.19.1 'numpy<2' --no-cache-dir

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
