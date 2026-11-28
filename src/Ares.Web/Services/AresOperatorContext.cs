using System.Collections.Concurrent;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.Extensions.Caching.Distributed;

namespace Ares.Web.Services;

/// <summary>
/// Per-request CSRF antiforgery state propagated from SSR App shell into interactive components.
/// </summary>
public sealed record AresAntiforgeryState(string FieldName, string RequestToken);

/// <summary>
/// Abstraction over server-side operator session persistence.
/// </summary>
public interface IAresSessionStore
{
    void Store(AresOperatorSession session);
    bool TryGet(string? sessionId, out AresOperatorSession? session);
    bool Remove(string? sessionId);
    AresOperatorSession? UpdateApiKey(string? sessionId, string newApiKey);
}

/// <summary>
/// Server-side authenticated operator session storing role, tenant scope, and
/// operator-scoped API key (never serialized into client-readable browser storage).
/// </summary>
public sealed record AresOperatorSession(
    string SessionId,
    string Email,
    string DisplayName,
    string Initials,
    string Role,
    string? OrganizationId,
    string EffectiveApiKey,
    DateTimeOffset AuthenticatedAt
)
{
    public bool IsAdmin => string.Equals(Role, "Admin", StringComparison.OrdinalIgnoreCase);

    public string ScopeLabel => IsAdmin
        ? "Admin · All Tenants"
        : $"Tenant · {OrganizationId}";

    public bool CanAccessOrganization(string? orgId)
    {
        if (IsAdmin) return true;
        if (string.IsNullOrWhiteSpace(OrganizationId) || string.IsNullOrWhiteSpace(orgId)) return false;
        return string.Equals(OrganizationId, orgId.Trim(), StringComparison.OrdinalIgnoreCase);
    }
}

/// <summary>
/// Thread-safe server-side registry binding opaque cookie session IDs (`ares:sid`)
/// to operator credentials and tenant scopes, with optional DataProtection-encrypted
/// disk persistence and IDistributedCache support.
/// </summary>
public sealed class AresSessionRegistry : IAresSessionStore
{
    private const string ProtectorPurpose = "Ares.OperatorSessions.v1";
    private readonly ConcurrentDictionary<string, AresOperatorSession> _sessions = new();
    private readonly IDataProtector? _protector;
    private readonly string? _persistencePath;
    private readonly IDistributedCache? _distributedCache;
    private readonly TimeSpan _sessionLifetime;
    private readonly object _ioLock = new();

    public AresSessionRegistry()
        : this(dataProtectionProvider: null, persistencePath: null, distributedCache: null, sessionLifetime: null)
    {
    }

    public AresSessionRegistry(
        IDataProtectionProvider? dataProtectionProvider,
        string? persistencePath = null,
        IDistributedCache? distributedCache = null,
        TimeSpan? sessionLifetime = null)
    {
        _persistencePath = string.IsNullOrWhiteSpace(persistencePath) ? null : persistencePath.Trim();
        _distributedCache = distributedCache;
        _sessionLifetime = sessionLifetime ?? TimeSpan.FromHours(8);

        if ((_persistencePath is not null || _distributedCache is not null) && dataProtectionProvider is null)
        {
            throw new ArgumentNullException(
                nameof(dataProtectionProvider),
                "IDataProtectionProvider is required when persistent or distributed session storage is enabled."
            );
        }

        _protector = dataProtectionProvider?.CreateProtector(ProtectorPurpose);
        if (_distributedCache is null && _persistencePath is not null)
        {
            LoadFromDisk();
        }
    }

    public void Store(AresOperatorSession session)
    {
        if (_distributedCache is not null && _protector is not null)
        {
            PersistSessionInCache(session);
            _sessions.TryRemove(session.SessionId, out _);
            return;
        }
        _sessions[session.SessionId] = session;
        SaveToDisk();
    }

    public bool TryGet(string? sessionId, out AresOperatorSession? session)
    {
        if (string.IsNullOrWhiteSpace(sessionId))
        {
            session = null;
            return false;
        }

        if (_distributedCache is not null && _protector is not null)
        {
            try
            {
                var protectedPayload = _distributedCache.GetString(GetCacheKey(sessionId));
                if (!string.IsNullOrWhiteSpace(protectedPayload))
                {
                    var json = _protector.Unprotect(protectedPayload);
                    var restored = JsonSerializer.Deserialize<AresOperatorSession>(json);
                    if (restored is not null && !IsExpired(restored))
                    {
                        session = restored;
                        return true;
                    }
                    _distributedCache.Remove(GetCacheKey(sessionId));
                }
            }
            catch
            {
                // Fail closed if distributed cache entry is corrupt, unreadable, or keys rotated.
            }

            session = null;
            return false;
        }

        if (_sessions.TryGetValue(sessionId, out var existing))
        {
            if (!IsExpired(existing))
            {
                session = existing;
                return true;
            }

            Remove(sessionId);
        }

        session = null;
        return false;
    }

    public bool Remove(string? sessionId)
    {
        if (string.IsNullOrWhiteSpace(sessionId)) return false;
        var localRemoved = _sessions.TryRemove(sessionId, out _);
        if (_distributedCache is not null)
        {
            try
            {
                var key = GetCacheKey(sessionId);
                var existed = !string.IsNullOrWhiteSpace(_distributedCache.GetString(key));
                _distributedCache.Remove(key);
                return existed || localRemoved;
            }
            catch
            {
                return localRemoved;
            }
        }
        SaveToDisk();
        return localRemoved;
    }

    public AresOperatorSession? UpdateApiKey(string? sessionId, string newApiKey)
    {
        if (string.IsNullOrWhiteSpace(sessionId) || string.IsNullOrWhiteSpace(newApiKey)) return null;
        if (TryGet(sessionId, out var existing) && existing is not null)
        {
            var updated = existing with { EffectiveApiKey = newApiKey.Trim() };
            Store(updated);
            return updated;
        }
        return null;
    }

    private bool IsExpired(AresOperatorSession session)
        => (DateTimeOffset.UtcNow - session.AuthenticatedAt) > _sessionLifetime;

    private static string GetCacheKey(string sessionId) => $"ares:session:{sessionId}";

    private void PersistSessionInCache(AresOperatorSession session)
    {
        if (_distributedCache is null || _protector is null) return;
        try
        {
            var protectedPayload = _protector.Protect(JsonSerializer.Serialize(session));
            _distributedCache.SetString(
                GetCacheKey(session.SessionId),
                protectedPayload,
                new DistributedCacheEntryOptions { SlidingExpiration = _sessionLifetime }
            );
        }
        catch
        {
            // Keep in-memory entry authoritative if transient cache write fails.
        }
    }

    private void LoadFromDisk()
    {
        if (_persistencePath is null || _protector is null) return;
        lock (_ioLock)
        {
            try
            {
                if (!File.Exists(_persistencePath)) return;
                var protectedBlob = File.ReadAllText(_persistencePath);
                if (string.IsNullOrWhiteSpace(protectedBlob)) return;
                var json = _protector.Unprotect(protectedBlob);
                var loaded = JsonSerializer.Deserialize<Dictionary<string, AresOperatorSession>>(json);
                if (loaded is null) return;
                foreach (var kv in loaded)
                {
                    if (!string.IsNullOrWhiteSpace(kv.Key) && kv.Value is not null && !IsExpired(kv.Value))
                    {
                        _sessions[kv.Key] = kv.Value;
                    }
                }
            }
            catch (Exception ex) when (ex is CryptographicException or JsonException or IOException)
            {
                // Unreadable or key-rotated session store fails closed; operators cleanly re-authenticate.
            }
        }
    }

    private void SaveToDisk()
    {
        if (_persistencePath is null || _protector is null) return;
        lock (_ioLock)
        {
            var dir = Path.GetDirectoryName(_persistencePath);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }

            var snapshot = new Dictionary<string, AresOperatorSession>();
            foreach (var kv in _sessions)
            {
                if (!IsExpired(kv.Value))
                {
                    snapshot[kv.Key] = kv.Value;
                }
            }

            var protectedBlob = _protector.Protect(JsonSerializer.Serialize(snapshot));
            var tmpPath = $"{_persistencePath}.{Guid.NewGuid():N}.tmp";
            File.WriteAllText(tmpPath, protectedBlob);
            if (!OperatingSystem.IsWindows())
            {
                try
                {
                    File.SetUnixFileMode(tmpPath, UnixFileMode.UserRead | UnixFileMode.UserWrite);
                }
                catch
                {
                    // Non-fatal on filesystems not supporting POSIX modes.
                }
            }
            File.Move(tmpPath, _persistencePath, overwrite: true);
        }
    }

    public static string ComputeInitials(string? displayName, string? email)
    {
        var source = !string.IsNullOrWhiteSpace(displayName) ? displayName.Trim() : (email ?? "OP").Trim();
        if (source.Contains('@'))
        {
            source = source.Split('@')[0];
        }
        var parts = source.Split([' ', '.', '_', '-'], StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length >= 2)
        {
            return $"{char.ToUpperInvariant(parts[0][0])}{char.ToUpperInvariant(parts[1][0])}";
        }
        if (source.Length >= 2)
        {
            return $"{char.ToUpperInvariant(source[0])}{char.ToUpperInvariant(source[1])}";
        }
        return source.Length == 1 ? $"{char.ToUpperInvariant(source[0])}A" : "OP";
    }
}

/// <summary>
/// Scoped operator context per Blazor circuit / HTTP request.
/// Resolves the authenticated operator's role, tenant `org_id`, and scoped API key
/// from ASP.NET Core cookie authentication and injects tenant headers onto backend calls.
/// </summary>
public sealed class AresOperatorContext
{
    public const string SessionIdClaim = "ares:sid";
    public const string OrgIdClaim = "ares:org_id";
    public const string InitialsClaim = "ares:initials";

    private readonly AuthenticationStateProvider? _authStateProvider;
    private readonly IHttpContextAccessor? _httpContextAccessor;
    private readonly AresSessionRegistry _registry;
    private AresOperatorSession? _overrideSession;

    public AresOperatorContext(
        AresSessionRegistry registry,
        AuthenticationStateProvider? authStateProvider = null,
        IHttpContextAccessor? httpContextAccessor = null)
    {
        _registry = registry;
        _authStateProvider = authStateProvider;
        _httpContextAccessor = httpContextAccessor;
    }

    /// <summary>Explicitly set session (useful in tests or synchronous circuit hydration).</summary>
    public void SetSession(AresOperatorSession? session) => _overrideSession = session;

    /// <summary>Synchronously resolve the active operator session if already hydrated or via HttpContext.</summary>
    public AresOperatorSession? Current
    {
        get
        {
            if (_overrideSession is not null) return _overrideSession;
            var user = _httpContextAccessor?.HttpContext?.User;
            return ResolveFromPrincipal(user);
        }
    }

    /// <summary>Asynchronously resolve and cache the active operator session from AuthenticationStateProvider.</summary>
    public async Task<AresOperatorSession?> GetSessionAsync()
    {
        if (_overrideSession is not null) return _overrideSession;

        ClaimsPrincipal? user = null;
        if (_authStateProvider is not null)
        {
            try
            {
                var state = await _authStateProvider.GetAuthenticationStateAsync();
                user = state.User;
            }
            catch
            {
                // Fallback to HttpContext below when outside an active Blazor circuit
            }
        }
        user ??= _httpContextAccessor?.HttpContext?.User;

        var resolved = ResolveFromPrincipal(user);
        if (resolved is not null)
        {
            _overrideSession = resolved;
        }
        return resolved;
    }

    public AresOperatorSession? ResolveFromPrincipal(ClaimsPrincipal? principal)
    {
        if (principal?.Identity?.IsAuthenticated != true) return null;
        var sid = principal.FindFirst(SessionIdClaim)?.Value;
        if (_registry.TryGet(sid, out var stored) && stored is not null)
        {
            return stored;
        }
        return null;
    }

    /// <summary>
    /// Apply operator-scoped authentication and tenant headers to an outgoing backend HTTP request,
    /// failing closed when no active operator session exists (never falling back to a global admin key).
    /// </summary>
    public void ApplyHeaders(HttpRequestMessage request)
    {
        ApplyHeadersFromSession(request, Current);
    }

    /// <summary>
    /// Asynchronously hydrate the active Blazor circuit/HTTP operator session and apply scoped headers.
    /// Fails closed if the session is missing or revoked.
    /// </summary>
    public async Task ApplyHeadersAsync(HttpRequestMessage request)
    {
        var session = await GetSessionAsync();
        ApplyHeadersFromSession(request, session);
    }

    private static void ApplyHeadersFromSession(HttpRequestMessage request, AresOperatorSession? session)
    {
        // Always strip pre-existing identity/tenant headers so no ambient/global admin key can leak.
        request.Headers.Remove("X-Ares-Api-Key");
        request.Headers.Remove("X-Ares-User-Email");
        request.Headers.Remove("X-Ares-Organization-Id");
        request.Headers.Remove("X-Ares-Org-Id");

        if (session is null || string.IsNullOrWhiteSpace(session.EffectiveApiKey))
        {
            throw new UnauthorizedAccessException(
                "No active ARES operator session is available. Please sign in again."
            );
        }

        request.Headers.TryAddWithoutValidation("X-Ares-Api-Key", session.EffectiveApiKey);

        if (!string.IsNullOrWhiteSpace(session.Email))
        {
            request.Headers.TryAddWithoutValidation("X-Ares-User-Email", session.Email);
        }

        if (!session.IsAdmin && !string.IsNullOrWhiteSpace(session.OrganizationId))
        {
            request.Headers.TryAddWithoutValidation("X-Ares-Organization-Id", session.OrganizationId);
            request.Headers.TryAddWithoutValidation("X-Ares-Org-Id", session.OrganizationId);
        }
    }

    /// <summary>Create an HttpClient scoped to the current authenticated operator's API key &amp; org headers.</summary>
    public HttpClient CreateScopedApiClient(
        IHttpClientFactory httpClientFactory,
        IConfiguration config,
        bool requireAuth = true)
    {
        return BuildScopedApiClient(httpClientFactory, config, Current, requireAuth);
    }

    /// <summary>Asynchronously hydrate the operator session and create a strictly scoped HttpClient.</summary>
    public async Task<HttpClient> CreateScopedApiClientAsync(
        IHttpClientFactory httpClientFactory,
        IConfiguration config,
        bool requireAuth = true)
    {
        var session = await GetSessionAsync();
        return BuildScopedApiClient(httpClientFactory, config, session, requireAuth);
    }

    private static HttpClient BuildScopedApiClient(
        IHttpClientFactory httpClientFactory,
        IConfiguration config,
        AresOperatorSession? session,
        bool requireAuth)
    {
        if (requireAuth && (session is null || string.IsNullOrWhiteSpace(session.EffectiveApiKey)))
        {
            throw new UnauthorizedAccessException(
                "No active ARES operator session is available. Please sign in again."
            );
        }

        var baseUrl = (config["Ares:ApiBaseUrl"] ?? "http://localhost:8000").TrimEnd('/') + "/";
        var http = httpClientFactory.CreateClient();
        http.BaseAddress = new Uri(baseUrl);

        if (session is not null && !string.IsNullOrWhiteSpace(session.EffectiveApiKey))
        {
            http.DefaultRequestHeaders.TryAddWithoutValidation("X-Ares-Api-Key", session.EffectiveApiKey);
            if (!string.IsNullOrWhiteSpace(session.Email))
            {
                http.DefaultRequestHeaders.TryAddWithoutValidation("X-Ares-User-Email", session.Email);
            }
            if (!session.IsAdmin && !string.IsNullOrWhiteSpace(session.OrganizationId))
            {
                http.DefaultRequestHeaders.TryAddWithoutValidation("X-Ares-Organization-Id", session.OrganizationId);
                http.DefaultRequestHeaders.TryAddWithoutValidation("X-Ares-Org-Id", session.OrganizationId);
            }
        }
        return http;
    }
}
