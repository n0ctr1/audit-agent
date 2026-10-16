# Semgrep with a Solidity-focused ruleset baked in for offline/network-isolated runs.
FROM semgrep/semgrep:latest

USER root

# Bake the Decurity smart-contract rules into the image at /rules so the wrapper
# can run `semgrep --config /rules` without any network access at audit time.
RUN apk add --no-cache git || (apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*)
RUN git clone --depth 1 https://github.com/Decurity/semgrep-smart-contracts /tmp/scrules \
    && mkdir -p /rules \
    && cp -r /tmp/scrules/solidity/. /rules/ 2>/dev/null || cp -r /tmp/scrules/. /rules/ \
    && rm -rf /tmp/scrules

WORKDIR /target

# Smoke-test.
RUN semgrep --version
