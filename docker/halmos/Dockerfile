# Halmos symbolic-execution image for the invariant-prover stage (roadmap #4).
# Build:  docker build -t audit-agent/halmos:latest docker/halmos
# The HalmosRunner degrades gracefully if this image is absent.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Foundry installed directly into /opt/foundry: foundryup links binaries via
# absolute symlinks into $FOUNDRY_DIR/versions, so moving the directory after
# installation leaves forge/cast dangling. Version pinned for reproducibility.
ARG FOUNDRY_VERSION=v1.8.5
ENV FOUNDRY_DIR=/opt/foundry
RUN curl -L https://foundry.paradigm.xyz | bash \
    && /opt/foundry/bin/foundryup --install "${FOUNDRY_VERSION}" \
    && /opt/foundry/bin/forge --version \
    && chmod -R a+rwX /opt/foundry
ENV PATH="/opt/foundry/bin:${PATH}"

# solc + halmos (ships with the z3 solver as a dependency).
RUN pip install --no-cache-dir solc-select halmos && \
    solc-select install 0.8.20 0.8.19 && \
    solc-select use 0.8.20

RUN mkdir -p /home/auditor && chmod 777 /home/auditor
ENV HOME=/home/auditor

WORKDIR /project
CMD ["/bin/bash"]
