# Build all services.
docker-build-all:
	docker-compose up --build --detach ;

# Turn-off all services.
docker-down-all:
	docker-compose down --volumes ;

# See logs for all services.
docker-logs:
	docker-compose logs --follow ;

# Run jupyter lab.
docker-jupyterlab:
	docker-compose exec statapp_service sh -c "jupyter-lab --allow-root --ip=0.0.0.0"

# Generate requirements.txt file without conda dependencies.
pip-dependencies:
	pip freeze | grep -v "@" > requirements.txt ;
