FROM python:3.13

ENV MPLBACKEND=Agg
ENV UV_LINK_MODE=copy

WORKDIR /photoz_estimation

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

CMD ["uv", "run", "bash", "run_fast_test.sh"]
