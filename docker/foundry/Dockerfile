FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Foundry
RUN curl -L https://foundry.paradigm.xyz | bash
ENV PATH="/root/.foundry/bin:${PATH}"
RUN foundryup

# Pre-initialise a template project so forge-std is cached in the image.
# At runtime, tests copy lib/ from here — no network access needed.
WORKDIR /forge-template
RUN forge init --no-git /forge-template

# Pre-compile the template to download and cache solc for 0.8.13 (default Counter pragma).
# Then add a 0.8.20 dummy and compile again to cache that compiler too.
RUN cd /forge-template && forge build
RUN printf '// SPDX-License-Identifier: MIT\npragma solidity ^0.8.20;\ncontract _D {}\n' \
        > /forge-template/src/Dummy820.sol && \
    cd /forge-template && forge build && \
    rm /forge-template/src/Dummy820.sol

# Use bash as entrypoint so docker run receives a shell command string
WORKDIR /project
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["forge --help"]
