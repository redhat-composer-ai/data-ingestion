# Use a Python image with uv pre-installed
FROM registry.access.redhat.com/ubi9/python-311:latest

# Install UV for package management
RUN pip install uv --no-cache-dir

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

COPY uv.lock .
COPY pyproject.toml .

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /opt/app-root/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/opt/app-root/src/.venv/bin:$PATH"
