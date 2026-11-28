# ARES Deployment Hardening & Operations Runbook

## 1. Production Baseline Architecture

Use [docker-compose.yml](file:///c:/Users/HP/Desktop/projects/ARES-AI-Resilience-and-Evaluation-Security-System-/docker-compose.yml) as the production deployment manifest. Do **not** deploy `docker-compose.full.yml` in production; it is strictly for local full-stack development (`mock` target apps, LoRA server, and non-password-protected dev Redis/Qdrant convenience).

Hardened guarantees in `docker-compose.yml`:
- Every host-published port (`5432`, `8000`, `5149`, and `9090`/`3000` under `--profile monitoring`) binds strictly to `127.0.0.1`.
- Redis (`--requirepass`) and Qdrant are internal-only on the Docker network with no host port exposure.
- `api` runs with `ENVIRONMENT=production`, `ARES_AUTH_ENABLED=true`, `ALLOW_DEV_IDENTITY=false`, `AUTO_CREATE_SCHEMA=false`, `GATEWAY_FAIL_CLOSED_ON_OUTAGE=true`, and `HMAC_ENFORCE_MODE=enforcing_critical`.
- `web` (Blazor Server) enforces Redis-backed multi-instance sessions (`Ares__RedisConnection`) and shared Redis ASP.NET DataProtection key rings.
- Grafana (`--profile monitoring`) has **no** default `admin` password (`GRAFANA_ADMIN_PASSWORD` is mandatory).

---

## 2. TLS & Reverse Proxy Setup

Always terminate external HTTPS traffic in front of `127.0.0.1:5149` (Blazor UI) and `127.0.0.1:8000` (FastAPI API / OpenAI reverse proxy). Never set `Ares__RequireHttpsCookie=false` on public HTTPS origins.

### Nginx Example (`/etc/nginx/conf.d/ares.conf`)
```nginx
server {
    listen 443 ssl http2;
    server_name ares.example.com;

    ssl_certificate     /etc/letsencrypt/live/ares.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ares.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /v1/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off; # Preserve SSE streaming
    }

    location / {
        proxy_pass http://127.0.0.1:5149;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade"; # Blazor SignalR
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Caddy Example (`Caddyfile`)
```caddy
ares.example.com {
    handle /api/* {
        reverse_proxy 127.0.0.1:8000
    }
    handle /v1/* {
        reverse_proxy 127.0.0.1:8000
    }
    handle {
        reverse_proxy 127.0.0.1:5149
    }
}
```

---

## 3. Mandatory Pre-Launch Secrets

Generate distinct high-entropy values before starting `docker compose up -d`:
```bash
export POSTGRES_PASSWORD=$(openssl rand -hex 24)
export REDIS_PASSWORD=$(openssl rand -hex 24)
export ARES_API_KEY="ares_$(openssl rand -hex 24)"
export HMAC_SECRET="$(openssl rand -hex 32)"
export PROMPT_FINGERPRINT_SECRET="$(openssl rand -hex 32)"
export GATEWAY_CANARY_TOKEN="CANARY_$(openssl rand -hex 16)"
export EXECUTION_PAYLOAD_ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
export GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 18)
```

---

## 4. Backups, Restore & Migration/Rollback Drill

### PostgreSQL Backup & Restore
```bash
# Backup (timestamped custom archive)
docker compose exec -T postgres pg_dump -U ares -Fc ares > "ares_$(date +%Y%m%d_%H%M%S).dump"

# Restore into clean DB
docker compose stop api web
docker compose exec -T postgres pg_restore -U ares -d ares --clean --if-exists < ares_backup.dump
docker compose start api web
```

### Redis AOF Snapshot Backup
```bash
docker compose exec -T redis sh -c 'REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli BGSAVE'
docker compose cp redis:/data/appendonlydir ./redis_backup_$(date +%Y%m%d)
```

### Alembic Migration & Rollback Drill
Verify schema migrations on staging prior to every production release:
```bash
docker compose exec api alembic current
docker compose exec api alembic upgrade head
# Rollback drill (one revision back, then forward again)
docker compose exec api alembic downgrade -1
docker compose exec api alembic upgrade head
```

---

## 5. Observability & Incident Response Runbook

1. **Enable Monitoring Stack**:
   ```bash
   GRAFANA_ADMIN_PASSWORD=... docker compose --profile monitoring up -d
   ```
   Alert rules in `monitoring/alerts.yml` monitor:
   - `AresFastApiDown` (scrape failure > 1m)
   - `AresGatewayHighLatencyP99` (p99 > 500ms for 5m)
   - `AresGatewayBlockSpike` (> 5 blocks/s for 2m)
   - `AresLlmFailureRateHigh` (> 25% upstream LLM error rate over 5m)

2. **Suspected Canary Leak or Credential Exposure**:
   - Immediately revoke compromised tenant key (`POST /api/v1/clients/{org_id}/api-key`) or rotate `ARES_API_KEY` + `HMAC_SECRET` + `GATEWAY_CANARY_TOKEN`.
   - Restart `prometheus` so `/tmp/ares_metrics_token` updates with the rotated `ARES_API_KEY`.
   - Force-evict active Blazor sessions by flushing Redis `ares:web:*` keys (`REDISCLI_AUTH="$REDIS_PASSWORD" redis-cli --scan --pattern "ares:web:*" | xargs -r ... DEL`).

3. **Vector Store Outage (`QDRANT_URL` unreachable)**:
   - In production (`GATEWAY_FAIL_CLOSED_ON_OUTAGE=true`), `/api/gateway/enforce` quarantines traffic cleanly once `qdrant_vector_store` trips instead of silently running without vector protection.
   - Verify Qdrant health (`docker compose ps qdrant`) and check `/healthz`; traffic automatically recovers once Qdrant passes health probes and the 30s half-open circuit-breaker window closes.
