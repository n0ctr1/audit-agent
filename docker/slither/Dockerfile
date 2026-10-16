FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget build-essential ca-certificates \
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

# Install Slither and solc-select
RUN pip install --no-cache-dir solc-select slither-analyzer

# Install compilers and make them available to all users
RUN solc-select install 0.8.20 0.8.19 0.8.0 && \
    solc-select use 0.8.20 && \
    chmod -R 777 /usr/local/bin && \
    chmod -R 777 /usr/local/lib/python3.11

# Create a home directory for the user (so solc-select does not complain)
RUN mkdir -p /home/auditor && chmod 777 /home/auditor
ENV HOME=/home/auditor

WORKDIR /target
CMD ["/bin/bash"]
