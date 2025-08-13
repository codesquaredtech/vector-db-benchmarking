echo "Creating network"
docker network create vector_db_testing

MODEL=vggface \
DIR="./images/NORTHSTORM/2024" \
docker compose -f "./vectorizer/docker-compose.yaml" up -d