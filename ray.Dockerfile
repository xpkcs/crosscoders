ARG RAY_VERSION="latest"
ARG ACCELERATOR="cpu"
FROM rayproject/ray:${RAY_VERSION}-py312-${ACCELERATOR}
ARG ACCELERATOR="cpu"

# -------------------------------------------------- #

ARG WORKDIR="/app"
WORKDIR "${WORKDIR}"


# # --- Create Conda Environment ---
# ARG PYTHON_VERSION="3.12.0"
# RUN conda create -n crosscoders "python~=${PYTHON_VERSION}"


# --- Dependencies ---
COPY --chown=ray:root README.md .
COPY --chown=ray:root pyproject.toml .
COPY --chown=ray:root requirements ./requirements

# --- Application Code ---
COPY --chown=ray:root src ./src


# --- Install Dependencies ---
RUN conda run -n base python -m pip install -r ./requirements/core.txt
RUN conda run -n base python -m pip install -r ./requirements/data.txt

RUN if [[ "${ACCELERATOR}" == "gpu" ]]; then \
        conda run -n base python -m pip install -r ./requirements/torch.txt; \
    fi

RUN conda run -n base python -m pip install .

# ARG XC_PKG_EXTRAS="ray-${ACCELERATOR}"
# ARG PIP_ARGS=""
# RUN echo "${ACCELERATOR} | ${XC_PKG_EXTRAS}"
# RUN [[ -z "${XC_PKG_EXTRAS}" ]] && PIP_PKG="." || PIP_PKG=".[${XC_PKG_EXTRAS}]" && \
#     PIP_CMD="pip install ${PIP_PKG} ${PIP_ARGS}" && \
#     conda run -n base python -m ${PIP_CMD}


# --- Environment Variables ---
ENV PYTHONUNBUFFERED 1
ENV CONFIG_PATH ./src/configs

EXPOSE 6379 8265 8080 3000