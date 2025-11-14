# SACP Suite Starter (Monorepo)

A ready-to-push starter repository that unifies **SACP**, **BCP**, **ABTC**, **MAESlab**, **Coupled Lorenz**, and **Hedrons (Attractorhedron/Inferencehedron)** into a single, production-oriented suite.

- **Backend API:** FastAPI (unified, versioned API across modules)
- **UI:** Dash multi-page app (clean navigation, per-module pages)
- **Core:** Plugin architecture (dynamics, metrics, controllers), Attractorhedron operator service, ABTC stubs
- **Cloud:** Docker Compose + Make targets; stateless API; file uploads via local disk (S3-ready)
- **Local Dev:** `uv`/pip, Python 3.11+

> This repo is designed to be **private**. It exposes capabilities via an **API**; distribution of binaries is not required.

## Quick start

```bash
# 0) (Optional) Create and activate a venv
python -m venv .venv && source .venv/bin/activate

# 1) Install
pip install -U pip
pip install -e ".[all]"

# 2) Run API
sacp-suite-api  # or: uvicorn sacp_suite.api.main:app --reload

# 3) Run UI (separate terminal)
sacp-suite-ui

# 4) Open
# API: http://127.0.0.1:8000/docs
# UI : http://127.0.0.1:8050
```

### Monorepo layout

```
sacp-suite-starter/
├── src/sacp_suite/                 # Python package (API + UI + core + modules)
│   ├── api/                        # FastAPI endpoints (unified suite API)
│   ├── core/                       # Plugin base classes + registry
│   ├── modules/                    # Integrated modules
│   │   ├── sacp_x/                 # SACP-X simulation + metrics
│   │   ├── attractorhedron/        # Data-driven operator + gates + escape
│   │   ├── bcp/                    # Bioelectric Control Panel (data interface stub)
│   │   ├── abtc/                   # Attractor-Based Trajectory Calculator (stubs)
│   │   └── maes/                   # MAES/ST placeholders (future)
│   └── ui/                         # Dash multi-page application
├── tests/                          # Smoke tests
├── docker-compose.yml              # Cloud-friendly dev deployment
├── Makefile                        # Developer commands
├── .env.example                    # Config
├── .gitignore
├── LICENSE                         # Private, non-redistributable
└── pyproject.toml                  # Build + deps + console scripts
```

### Why this structure?

The plugin registry lets you drop new systems without touching the core.

The API is the product surface (private or public).

The UI consumes the same API, so it can be hosted or bundled for desktop use.

### Security defaults

- API binds to 127.0.0.1 by default; set `SACP_BIND=0.0.0.0` to expose.
- CORS is off by default; enable allowed origins via `SACP_CORS`.
- File uploads land in `./var/uploads` (git-ignored). Swap for S3 by replacing the storage adapter.

### Push this to your private Git host

```
git init
git add .
git commit -m "feat: SACP Suite Starter (API+UI+Core)"
git remote add origin <your-private-remote-url>
git branch -M main
git push -u origin main
```

Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
