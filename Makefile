# Build all services.
docker-build-all:
	docker-compose up --build --detach ;

# Turn-off all services.
docker-down-all:
	docker-compose down --volumes ;

# See logs for all services.
docker-logs:
			docker-compose logs --follow ;

