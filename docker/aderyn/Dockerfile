# Aderyn — Cyfrin's Rust/AST-based Solidity static analyzer.
# Uses the official prebuilt linux binary (parses .sol directly, no solc/foundry needed).
FROM debian:bookworm-slim

# Bump as new Aderyn releases land.
ARG ADERYN_VERSION=0.6.8

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/aderyn.tar.xz \
        "https://github.com/Cyfrin/aderyn/releases/download/aderyn-v${ADERYN_VERSION}/aderyn-x86_64-unknown-linux-gnu.tar.xz" \
    && mkdir -p /tmp/ad \
    && tar -xJf /tmp/aderyn.tar.xz -C /tmp/ad \
    && find /tmp/ad -type f -name aderyn -exec cp {} /usr/local/bin/aderyn \; \
    && chmod +x /usr/local/bin/aderyn \
    && rm -rf /tmp/aderyn.tar.xz /tmp/ad

WORKDIR /target

# Smoke-test: aderyn must report its version.
RUN aderyn --version

CMD ["/bin/bash"]
