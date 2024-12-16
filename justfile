compose := "docker compose"
image := "ghcr.io/aidotse/docling-inference"
rev := shell("git rev-parse --short HEAD")

default:
	just --list

build:
	docker build -t {{image}}:dev -t {{image}}:{{rev}} .

build-cpu:
	docker build -f Dockerfile.cpu -t {{image}}:cpu-dev -t {{image}}:cpu-{{rev}} .

up:
	{{compose}} up --remove-orphans

fmt:
	ruff format .
	ruff --fix .
