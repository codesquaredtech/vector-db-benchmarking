echo "Creating network"
docker network create vector_db_testing

echo "Starting milvus"
docker compose -f "./milvus/docker-compose.yaml" up -d

sleep 10

echo "Starting vectorizer"
docker compose -f "./vectorizer/docker-compose.yaml" up -d