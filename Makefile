.PHONY: all setup install_requirements download_data requirements docs

# Set name of virtual environment here
VENV = .venv
BIN = ${VENV}/bin/

all: setup docs

setup: ${VENV} install_requirements

${VENV}:
	python3 -m venv $@

install_requirements: requirements.txt
	$(BIN)pip install --upgrade pip
	$(BIN)pip install -r requirements.txt
	$(BIN)pre-commit install
	sudo apt-get install python3-tk -y

requirements:
	pip freeze > requirements.txt

