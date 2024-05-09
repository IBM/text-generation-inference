ARG BASE_UBI_IMAGE_TAG=9.3-1552
FROM registry.access.redhat.com/ubi9/ubi:${BASE_UBI_IMAGE_TAG}

RUN dnf update -y && dnf install -y python3-pip python3-wheel git && dnf clean all

WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY tgis_eval tgis_eval

RUN pip install .

RUN useradd app
VOLUME /cache
ENV XDG_CACHE_HOME=/cache

USER app
