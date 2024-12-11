compose := "docker compose"
image := "docling-inference"

default:
	just --list

build:
	docker build -t {{image}}:dev .

up:
	{{compose}} up -d --remove-orphans

fmt:
	ruff format .
	ruff --fix .
