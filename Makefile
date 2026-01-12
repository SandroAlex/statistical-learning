# Makefile for managing Docker services, Python dependencies, and Mlflow
.PHONY: help build up down debug jupyter requirements requirements-dev lock update mlflow-ui mlflow-clean

# Suppress make's default output
MAKEFLAGS += --no-print-directory

# Colorful output in terminal
GREEN=\033[0;32m
BLUE=\033[0;34m
YELLOW=\033[0;33m
RED=\033[0;31m
NC=\033[0;0m # No Color

# Mlflow configuration
MLFLOW_URI=sqlite:////statapp/mlflow/database.db
MLFLOW_PORT=5005

# Jupyterlab configuration
JUPYTER_PORT=8874

##### Help message #####
help: # Show this help message
	@printf "\n" ;
	@printf "${GREEN}Available make commands:${NC}\n" ;
	@printf "${YELLOW}  build${NC}           - Build docker image for all services\n" ;
	@printf "${YELLOW}  up${NC}              - Start all docker services\n" ;
	@printf "${YELLOW}  down${NC}            - Stop all docker services\n" ;
	@printf "${YELLOW}  debug${NC}           - Run a bash terminal inside a docker container\n" ;
	@printf "${YELLOW}  jupyter${NC}         - Run jupyterlab on statapp service\n" ;
	@printf "${YELLOW}  requirements${NC}    - Create requirements.txt file\n" ;
	@printf "${YELLOW}  lock${NC}            - Create / update uv.lock file\n" ;
	@printf "${YELLOW}  update${NC}          - Run all uv update commands\n" ;
	@printf "${YELLOW}  mlflow-ui${NC}       - Run Mlflow Tracking User Interface\n" ;
	@printf "${YELLOW}  mlflow-clean${NC}    - Cleanup Mlflow database and artifacts\n" ;
	@printf "\n" ;	


##### Docker commands #####
build: # Build docker image for all services
	@printf "\n" ;
	@printf "${GREEN}Building services ...${NC}\n" ;
	@docker-compose build ;
	@printf "${BLUE}Docker images built successfully!${NC}\n" ;
	@printf "\n" ; 

up: # Start all docker services
	@printf "\n" ;
	@printf "${GREEN}Starting all docker services ...${NC}\n" ;
	@docker-compose up --detach ;
	@printf "${BLUE}All docker services started successfully!${NC}\n" ;
	@printf "\n" ;

down: # Stop all docker services
	@printf "\n" ;
	@printf "${GREEN}Stopping all docker services ...${NC}\n" ;
	@docker-compose down ;
	@printf "${BLUE}All docker services stopped successfully!${NC}\n" ;
	@printf "\n" ;

debug: # Run a bash terminal inside a docker container
	@printf "\n" ;
	@printf "${GREEN}Starting bash terminal inside docker container ...${NC}\n" ;
	@docker-compose exec statapp_service sh -c "/bin/bash" ;
	@printf "${BLUE}Exited bash terminal inside docker container!${NC}\n" ;
	@printf "\n" ;

jupyter: # Run jupyterlab on statapp service
	@printf "\n" ;
	@printf "${GREEN}Starting JupyterLab inside docker container ...${NC}\n" ;
	@printf "${BLUE}Access link will be shown below:${NC}\n" ;
	@docker-compose exec statapp_service sh -c "uv run jupyter lab --ip=0.0.0.0 --port ${JUPYTER_PORT} --allow-root" || true ;
	@printf "${BLUE}JupyterLab session ended inside docker container!${NC}\n" ;
	@printf "\n" ;


##### Python dependencies management with uv #####
requirements: # Create requirements.txt file
	@printf "\n" ;
	@printf "${GREEN}Generating requirements.txt file ...${NC}\n" ;
	@uv export --format requirements-txt --no-hashes > requirements.txt ;
	@printf "${BLUE}requirements.txt file generated successfully!${NC}\n" ;
	@printf "\n" ;

lock: # Create / update uv.lock file
	@printf "\n" ;
	@printf "${GREEN}Updating uv.lock file ...${NC}\n" ;
	@uv lock ;
	@printf "${BLUE}uv.lock file updated successfully!${NC}\n" ;
	@printf "\n" ;

update: # Run all uv update commands
	@printf "\n" ;
	@printf "${GREEN}Updating uv.lock and requirement files ...${NC}\n" ;
	@$(MAKE) lock ;
	@$(MAKE) requirements ;
	@printf "${BLUE}uv.lock and requirement files updated successfully!${NC}\n" ;
	@printf "\n" ;


##### Mlflow #####
mlflow-ui: # Run Mlflow Tracking User Interface
	@printf "\n" ;
	@printf "${GREEN}Starting Mlflow Tracking User Interface ...${NC}\n" ;
	@uv run mlflow server --backend-store-uri ${MLFLOW_URI} --host 0.0.0.0 --port ${MLFLOW_PORT};
	@printf "${BLUE}Mlflow Tracking User Interface stopped!${NC}\n" ;
	@printf "\n" ;

mlflow-clean: # Cleanup Mlflow database and artifacts
	@printf "\n" ;
	@printf "${RED}Cleaning up Mlflow database and runs ...${NC}\n" ;
	@rm -f /statapp/mlflow/database.db ;
	@rm -rf /statapp/mlflow/mlruns ;
	@printf "${BLUE}Mlflow database and runs cleaned up successfully!${NC}\n" ;
	@printf "\n" ;
