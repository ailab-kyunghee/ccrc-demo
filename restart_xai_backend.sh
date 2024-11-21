# restart_xai_backend.sh
#!/bin/bash

# Restart the xai-backend-api container
docker container restart xai-backend-api

# Prune unused Docker images
docker image prune -f
