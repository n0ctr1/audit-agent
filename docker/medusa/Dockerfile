# Medusa — Trail of Bits' parallelized, coverage-guided Solidity fuzzer.
# Consumes the same property contracts as Echidna via crytic-compile.
FROM python:3.11-slim

# Bump as new Medusa releases land.
ARG MEDUSA_VERSION=1.5.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl wget ca-certificates unzip tar git \
    && rm -rf /var/lib/apt/lists/*

# crytic-compile + solc so Medusa can compile contracts.
RUN pip install --no-cache-dir crytic-compile solc-select \
    && solc-select install 0.8.20 \
    && solc-select use 0.8.20

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

# Medusa precompiled binary.
RUN curl -fsSL -o /tmp/medusa.tar.gz \
        "https://github.com/crytic/medusa/releases/download/v${MEDUSA_VERSION}/medusa-linux-x64.tar.gz" \
    && tar -xzf /tmp/medusa.tar.gz -C /usr/local/bin medusa \
    && chmod +x /usr/local/bin/medusa \
    && rm /tmp/medusa.tar.gz

RUN mkdir -p /home/auditor && chmod 777 /home/auditor
ENV HOME=/home/auditor

WORKDIR /workspace

# Smoke-test.
RUN medusa --version && crytic-compile --version
