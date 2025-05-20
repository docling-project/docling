# Build stage for installing dependencies
FROM node:20-slim AS builder

# Install necessary tools and prepare dependencies environment in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /build/bin /build/lib/node_modules \
    && cp /usr/local/bin/node /build/bin/    
# Set working directory
WORKDIR /build

# Create package.json and install Apify CLI in one layer
RUN echo '{"name":"docling-actor-dependencies","version":"1.0.0","description":"Dependencies for Docling Actor","private":true,"type":"module","engines":{"node":">=18"}}' > package.json \
    && npm install apify-cli@latest \
    && cp -r node_modules/* lib/node_modules/ \
    && echo '#!/bin/sh\n/tmp/docling-tools/bin/node /tmp/docling-tools/lib/node_modules/apify-cli/bin/run "$@"' > bin/actor \
    && chmod +x bin/actor \
    # Clean up npm cache to reduce image size
    && npm cache clean --force

# Final stage with docling-serve-cpu
FROM quay.io/ds4sd/docling-serve-cpu:latest

LABEL maintainer="Vaclav Vancura <@vancura>" \
      description="Apify Actor for document processing using Docling" \
      version="1.1.0"

# Set only essential environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCLING_SERVE_HOST=0.0.0.0 \
    DOCLING_SERVE_PORT=5001

# Switch to root temporarily to set up directories and permissions
USER root
WORKDIR /app

# Install required tools and create directories in a single layer
RUN dnf install -y \
    jq \
    && dnf clean all \
    && mkdir -p /build-files \
             /tmp \
             /tmp/actor-input \
             /tmp/actor-output \
             /tmp/actor-storage \
             /tmp/apify_input \
             /apify_input \
             /opt/app-root/src/.EasyOCR/user_network \
             /tmp/easyocr-models \
    && chown 1000:1000 /build-files \
    && chown -R 1000:1000 /opt/app-root/src/.EasyOCR \
    && chmod 1777 /tmp \
    && chmod 1777 /tmp/easyocr-models \
    && chmod 777 /tmp/actor-input /tmp/actor-output /tmp/actor-storage /tmp/apify_input /apify_input \
    # Fix for uv_os_get_passwd error in Node.js
    && echo "docling:x:1000:1000:Docling User:/app:/bin/sh" >> /etc/passwd

# Set environment variable to tell EasyOCR to use a writable location for models
ENV EASYOCR_MODULE_PATH=/tmp/easyocr-models

# Copy only required files
COPY --chown=1000:1000 .actor/actor.sh .actor/actor.sh
COPY --chown=1000:1000 .actor/actor.json .actor/actor.json
COPY --chown=1000:1000 .actor/input_schema.json .actor/input_schema.json
RUN chmod +x .actor/actor.sh

# Copy the build files from builder
COPY --from=builder --chown=1000:1000 /build /build-files


# Switch to non-root user
USER 1000

# Set up TMPFS for temporary files
VOLUME ["/tmp"]

# Create additional volumes for OCR models persistence
VOLUME ["/tmp/easyocr-models"]

# Expose the docling-serve API port
EXPOSE 5001

# Run the actor script
ENTRYPOINT [".actor/actor.sh"]
