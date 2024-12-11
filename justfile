compose := "docker compose"
image := "docling-inference"
rev := shell("git rev-parse --short HEAD")


default:
	just --list

build:
	docker build -t {{image}}:dev -t {{image}}:{{rev}} .

up:
	{{compose}} up --remove-orphans

fmt:
	ruff format .
	ruff --fix .
