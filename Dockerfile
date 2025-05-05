# Stage 1: Build dependencies
FROM python:3.11.8-slim as builder

# Install UV
RUN pip install --no-cache-dir uv

# Copy only the pyproject.toml file
COPY pyproject.toml ./

# Install dependencies only using UV
RUN uv pip install --system .

# Stage 2: Final image
FROM python:3.11.8-slim

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY ./data ./data
COPY ./models ./models
COPY ./visualizations ./visualizations
COPY ./run_training.py .
COPY ./run_testing.py .
COPY ./config.py .
COPY pyproject.toml ./
