.PHONY: install dev-up dev-down db-init neo4j-init qdrant-init build-slither build-foundry build-aderyn build-semgrep build-medusa build-halmos build-all lint test clean

install:
	uv sync

dev-up:
	docker compose up -d
	@echo "Waiting for services..."
	@sleep 5

dev-down:
	docker compose down

db-init:
	uv run audit-agent db-init

neo4j-init:
	uv run audit-agent neo4j-init

qdrant-init:
	uv run audit-agent qdrant-init

infra-init: db-init neo4j-init qdrant-init
	@echo "Infrastructure initialized."

build-slither:
	docker build -t audit-agent/slither:latest docker/slither/

build-foundry:
	docker build -t audit-agent/foundry:latest docker/foundry/

build-aderyn:
	docker build -t audit-agent/aderyn:latest docker/aderyn/

build-semgrep:
	docker build -t audit-agent/semgrep:latest docker/semgrep/

build-medusa:
	docker build -t audit-agent/medusa:latest docker/medusa/

build-halmos:
	docker build -t audit-agent/halmos:latest docker/halmos/

build-all: build-slither build-foundry build-aderyn build-semgrep build-medusa build-halmos

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fmt:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

test:
	uv run pytest tests/unit/ -v

test-all:
	uv run pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

# Quick smoke test (no infra required)
smoke:
	@mkdir -p /tmp/smoke-contracts
	@cp test-contracts/*.sol /tmp/smoke-contracts/ 2>/dev/null || true
	uv run audit-agent audit /tmp/smoke-contracts --mode review --log-level DEBUG
