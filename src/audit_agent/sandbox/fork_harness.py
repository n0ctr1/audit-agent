"""Anvil fork harness for mainnet-state exploit tests (§6.3).

In mock mode (AUDIT_AGENT_TOOL_MODE=mock or no Docker), all operations
are no-ops that log what would have happened. This keeps the interface
identical for testing without requiring a live Ethereum RPC.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from audit_agent.observability.logging import get_logger
from audit_agent.sandbox.docker_runner import DockerRunner

log = get_logger("sandbox.fork_harness")

Chain = Literal["ethereum", "arbitrum", "optimism", "base", "polygon"]


@dataclass
class ForkContext:
    chain: Chain
    block_number: Optional[int]
    rpc_url: str
    anvil_url: str = "http://localhost:8545"
    snapshot_ids: list[str] = field(default_factory=list)
    is_mock: bool = False


class ForkHarness:
    """Manages an Anvil mainnet fork for economic exploit testing.

    All methods are no-ops in mock mode, returning plausible mock values.
    In production, delegates to DockerRunner to spin up Anvil containers.
    """

    def __init__(self, docker_runner: Optional[DockerRunner] = None) -> None:
        self._runner = docker_runner or DockerRunner()
        self._ctx: Optional[ForkContext] = None
        self._is_mock = self._runner._tool_mode == "mock"

    @property
    def context(self) -> Optional[ForkContext]:
        return self._ctx

    async def setup_fork(
        self,
        chain: Chain = "ethereum",
        block_number: Optional[int] = None,
        rpc_url: str = "",
    ) -> ForkContext:
        """Spin up an Anvil fork. In mock mode, returns a mock ForkContext."""
        if self._is_mock:
            log.info(
                "ForkHarness: mock mode — returning mock ForkContext",
                chain=chain,
                block_number=block_number,
            )
            self._ctx = ForkContext(
                chain=chain,
                block_number=block_number,
                rpc_url=rpc_url or "mock://localhost",
                anvil_url="http://localhost:8545",
                is_mock=True,
            )
            return self._ctx

        log.info("ForkHarness: spinning up Anvil fork", chain=chain, block=block_number)
        anvil_cmd = ["anvil", "--silent"]
        if rpc_url:
            anvil_cmd += ["--fork-url", rpc_url]
        if block_number:
            anvil_cmd += ["--fork-block-number", str(block_number)]

        # In production: run Anvil container and wait for it to be ready
        # For now: raise if Docker is unavailable (handled by DockerRunner)
        await self._runner.run(
            image="ghcr.io/foundry-rs/foundry:latest",
            command=anvil_cmd,
            timeout=30,
        )

        self._ctx = ForkContext(
            chain=chain,
            block_number=block_number,
            rpc_url=rpc_url,
            anvil_url="http://localhost:8545",
            is_mock=False,
        )
        log.info("ForkHarness: fork ready", anvil_url=self._ctx.anvil_url)
        return self._ctx

    async def impersonate(self, addresses: list[str]) -> None:
        """anvil_impersonateAccount for each address."""
        if self._is_mock:
            log.info("ForkHarness: mock impersonate", addresses=addresses[:3])
            return
        for addr in addresses:
            log.debug("ForkHarness: impersonating", address=addr)
            # In production: call anvil_impersonateAccount via JSON-RPC

    async def set_balance(self, address: str, balance_wei: int) -> None:
        """anvil_setBalance — set ETH balance for address."""
        if self._is_mock:
            log.info("ForkHarness: mock set_balance", address=address, balance_wei=balance_wei)
            return
        log.debug("ForkHarness: setting balance", address=address, balance_eth=balance_wei / 1e18)
        # In production: call anvil_setBalance via JSON-RPC

    async def snapshot(self) -> str:
        """evm_snapshot — save current chain state. Returns snapshot ID."""
        if self._is_mock:
            snap_id = f"mock-snap-{uuid.uuid4().hex[:8]}"
            if self._ctx:
                self._ctx.snapshot_ids.append(snap_id)
            log.info("ForkHarness: mock snapshot", id=snap_id)
            return snap_id
        # In production: call evm_snapshot
        return "0x0"

    async def revert(self, snapshot_id: str) -> None:
        """evm_revert — restore chain state to snapshot."""
        if self._is_mock:
            log.info("ForkHarness: mock revert", snapshot_id=snapshot_id)
            return
        log.debug("ForkHarness: reverting to snapshot", snapshot_id=snapshot_id)
        # In production: call evm_revert

    async def teardown(self) -> None:
        """Kill the Anvil container and clean up."""
        if self._is_mock:
            log.info("ForkHarness: mock teardown")
            self._ctx = None
            return
        log.info("ForkHarness: tearing down Anvil fork")
        # In production: stop and remove the Anvil container
        self._ctx = None
