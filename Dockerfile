FROM python:3.12-slim

WORKDIR /app

# Debian security fixes newer than the base image. The base still ships libzstd1 without the
# CVE-2026-124851 fix (1.5.7+dfsg-1+deb13u1, trixie-security), so the build fails if the upgrade
# did not bring it in. Naming it also changes this layer, so a build cache from before the fix
# cannot be reused. Drop the check once python:3.12-slim ships the fixed version.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && dpkg --compare-versions "$(dpkg-query -W -f='${Version}' libzstd1)" ge 1.5.7+dfsg-1+deb13u1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --system --uid 65532 --no-create-home embeddings

COPY embeddings ./embeddings
USER 65532:65532

EXPOSE 8030
CMD ["python", "-m", "uvicorn", "embeddings.main:app", "--host", "0.0.0.0", "--port", "8030"]
