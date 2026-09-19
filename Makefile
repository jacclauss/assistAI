.DEFAULT_GOAL := help
COMPOSE := docker compose -f docker/docker-compose.yml
COMPOSE_DEV := $(COMPOSE) -f docker/compose.dev.yml

.PHONY: help
help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Install the toolchain and create the virtualenv
	command -v uv >/dev/null || brew install uv
	uv sync

.PHONY: lock
lock: ## Regenerate uv.lock with pinned versions and hashes
	uv lock

.PHONY: fmt
fmt: ## Format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: lint
lint: ## Lint without modifying files
	uv run ruff format --check .
	uv run ruff check .

.PHONY: typecheck
typecheck: ## Strict type check
	uv run mypy

.PHONY: test
test: ## Run the test suite
	uv run pytest -q

.PHONY: check
check: lint typecheck test ## Everything CI would run

.PHONY: run
run: ## Run the gateway on the host with console logs
	ASSISTAI_LOG_CONSOLE=true ASSISTAI_HEARTBEAT_SECONDS=5 uv run python -m assistai

.PHONY: chat
chat: ## Multi-turn Fireworks conversation on the terminal
	ASSISTAI_LOG_CONSOLE=true uv run python -m assistai chat

.PHONY: compare
compare: ## Score primary and candidate models against get_time
	ASSISTAI_LOG_CONSOLE=true uv run python -m assistai models compare

.PHONY: live
live: ## Run the optional live Fireworks tests (costs tokens)
	ASSISTAI_LIVE=1 uv run pytest -q -m live -o addopts=

.PHONY: build
build: ## Build the gateway image for this machine
	$(COMPOSE) build

.PHONY: up
up: ## Start the stack (no published ports)
	$(COMPOSE) up -d

.PHONY: up-dev
up-dev: ## Start the stack with signal-cli on 127.0.0.1:8080
	$(COMPOSE_DEV) up -d

.PHONY: down
down: ## Stop the stack
	$(COMPOSE) down

.PHONY: logs
logs: ## Follow gateway logs
	$(COMPOSE) logs -f gateway

.PHONY: logs-signal
logs-signal: ## Follow signal-cli logs
	$(COMPOSE) logs -f signal-cli

.PHONY: signal-link
signal-link: ## Print a device-link URI (scan from Signal on your phone)
	$(COMPOSE) run --rm --no-deps gateway signal link

.PHONY: signal-health
signal-health: ## Check signal-cli from inside the compose network
	$(COMPOSE) run --rm --no-deps gateway signal health

.PHONY: build-pi
build-pi: ## Build the arm64 image for the Raspberry Pi
	docker buildx build --platform linux/arm64 \
		-f docker/gateway.Dockerfile \
		-t assistai/gateway:local --load .

.PHONY: save-pi
save-pi: build-pi ## Export the arm64 image for transfer to the Pi
	docker save assistai/gateway:local -o assistai-gateway-arm64.tar
	@echo "Copy to the Pi, then: docker load -i assistai-gateway-arm64.tar"
