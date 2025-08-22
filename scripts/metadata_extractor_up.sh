echo "Creating network"
docker network create vector_db_testing

docker compose -f "./metadata_extractor/docker-compose.yaml" up -d --build