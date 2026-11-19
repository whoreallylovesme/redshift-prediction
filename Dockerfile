FROM python:3.13

ENV MPLBACKEND=Agg
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY lib/ lib/
COPY runtime/ runtime/
COPY tools/ tools/
COPY photoz/ photoz/
COPY configs/ configs/
COPY tests/fixtures/ tests/fixtures/
RUN uv sync --frozen

CMD ["uv", "run", "python", "-m", "runtime.pipeline", "--config", "configs/flows/regression_example.yaml"]
