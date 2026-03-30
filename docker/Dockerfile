# use debian based so it will include C++ build tools
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim  

# Install C++ build tools (for grakel)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
# Copy the project into the image
COPY westerfeld /app/repository/westerfeld
COPY pyproject.toml /app/repository

# replace ../ by /app/data/, so the data can be mounted easily
COPY . /app
RUN find /app -name "*.py" -exec sed -i -E 's|(\.\.\/)+|/app/input/|g' {} + \
    && grep -r "\.\." /app --include="*.py" || echo "All ../ replaced successfully"

# Disable development dependencies
ENV UV_NO_DEV=1

WORKDIR /app/repository
RUN ls
RUN uv sync

COPY docker/entrypoint.sh /app/repository/entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/app/repository/entrypoint.sh"]