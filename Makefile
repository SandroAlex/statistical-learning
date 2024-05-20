# Build all services.
docker-build-all:
	docker-compose up --build --detach ;

# Turn-off all services.
docker-down-all:
	docker-compose down --volumes ;

# See logs for all services.
docker-logs:
	docker-compose logs --follow ;

# Generate requirements.txt file without conda dependencies.
pip-dependencies:
	pip freeze | grep -v "@" > requirements.txt ;

# Run jupyter lab.
jupyterlab:
	jupyter-lab --allow-root --ip=0.0.0.0
