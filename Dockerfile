# Multi-stage build for ML-Based Network Intrusion Detection Systems
# Stage 1: Build the conda environment once.
FROM continuumio/miniconda3:latest AS builder

WORKDIR /build
COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda clean -afy

# Stage 2: Runtime image with project code and prebuilt environment.
FROM continuumio/miniconda3:latest

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/conda/envs/nids-adv/bin:/opt/conda/bin:$PATH

RUN useradd -m -u 1000 -s /bin/bash researcher

WORKDIR /workspace

# Copy prebuilt environment from builder image.
COPY --from=builder /opt/conda/envs/nids-adv /opt/conda/envs/nids-adv

# Copy full repository contents into the image.
COPY --chown=researcher:researcher . /workspace

USER researcher

EXPOSE 8888

# PATH already points to the conda env Python and tools.
CMD ["/bin/bash"]
