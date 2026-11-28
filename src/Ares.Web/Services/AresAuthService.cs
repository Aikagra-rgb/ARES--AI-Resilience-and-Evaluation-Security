using System.Net;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Authentication.Cookies;

namespace Ares.Web.Services;

public sealed record AresLoginRequest(
    string Email,
    string? DisplayName,
    string Role,
    string? OrganizationId,
    string ApiKey
);

public sealed record AresAuthResult(
    bool Succeeded,
    string? ErrorMessage = null,
    AresOperatorSession? Session = null,
    ClaimsPrincipal? Principal = null
);

/// <summary>
/// Authenticates Blazor console operators into either Admin (all tenants) or
/// Tenant-scoped (`org_id`) cookie sessions verified against configured keys or
/// the live FastAPI backend.
/// </summary>
public sealed class AresAuthService
{
    private readonly IConfiguration _config;
    private readonly AresSessionRegistry _registry;
    private readonly HttpClient _http;

    public AresAuthService(
        IConfiguration config,
        AresSessionRegistry registry,
        HttpClient? httpClient = null)
    {
        _config = config;
        _registry = registry;
        if (httpClient is not null)
        {
            _http = httpClient;
        }
        else
        {
            var baseUrl = (config["Ares:ApiBaseUrl"] ?? "http://localhost:8000").TrimEnd('/') + "/";
            _http = new HttpClient
            {
                BaseAddress = new Uri(baseUrl),
                Timeout = TimeSpan.FromSeconds(5),
            };
        }
    }

    public async Task<AresAuthResult> AuthenticateAsync(
        AresLoginRequest request,
        CancellationToken cancellationToken = default)
    {
        var email = (request.Email ?? string.Empty).Trim().ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(email) || !email.Contains('@') || email.Length > 320)
        {
            return new AresAuthResult(false, "Please enter a valid operator email address.");
        }

        var submittedKey = (request.ApiKey ?? string.Empty).Trim();
        if (string.IsNullOrWhiteSpace(submittedKey))
        {
            return new AresAuthResult(false, "An ARES Admin or Tenant-Scoped API Key is required.");
        }

        var roleNorm = string.Equals(request.Role?.Trim(), "Admin", StringComparison.OrdinalIgnoreCase)
            ? "Admin"
            : "Tenant";

        var configuredAdminKey = (_config["Ares:ApiKey"] ?? "ares-dev-secret-key-42").Trim();
        var isMasterKey = ConstantTimeEquals(submittedKey, configuredAdminKey);

        if (roleNorm == "Admin")
        {
            if (submittedKey.StartsWith("ares_tnt_", StringComparison.OrdinalIgnoreCase))
            {
                return new AresAuthResult(
                    false,
                    "Tenant-scoped API keys cannot sign in with the Admin role. Choose 'Tenant Operator'."
                );
            }

            if (!isMasterKey)
            {
                var backendAdminOk = await VerifyKeyAgainstBackendAsync(submittedKey, null, expectAdmin: true, cancellationToken);
                if (!backendAdminOk)
                {
                    return new AresAuthResult(false, "Invalid Admin API key.");
                }
            }

            return IssueSession(email, request.DisplayName, "Admin", null, submittedKey);
        }

        // Tenant-scoped operator authentication
        var orgId = (request.OrganizationId ?? string.Empty).Trim();
        if (string.IsNullOrWhiteSpace(orgId) || string.Equals(orgId, "all", StringComparison.OrdinalIgnoreCase))
        {
            return new AresAuthResult(false, "Select or enter a specific Organization ID for Tenant Operator access.");
        }

        // Reject keys explicitly minted for a different tenant prefix
        if (submittedKey.StartsWith("ares_tnt_", StringComparison.OrdinalIgnoreCase) &&
            !submittedKey.StartsWith($"ares_tnt_{orgId}_", StringComparison.OrdinalIgnoreCase))
        {
            return new AresAuthResult(
                false,
                $"API key belongs to a different organization and cannot authenticate into '{orgId}'."
            );
        }

        var tenantKeys = ParseConfiguredTenantKeys();
        var matchesConfiguredTenantKey =
            tenantKeys.TryGetValue(orgId, out var expectedTenantKey) &&
            ConstantTimeEquals(submittedKey, expectedTenantKey);

        if (!matchesConfiguredTenantKey && !isMasterKey)
        {
            var (reachedBackend, authorized) = await VerifyTenantKeyWithStatusAsync(submittedKey, orgId, cancellationToken);
            if (reachedBackend)
            {
                if (!authorized)
                {
                    return new AresAuthResult(
                        false,
                        $"Access denied: API key is not authorized for organization '{orgId}'."
                    );
                }
            }
            else
            {
                // Backend unreachable: only permit offline mock mode if explicitly enabled and matching tenant key prefix
                var useMock = _config.GetValue<bool>("Ares:UseMock", defaultValue: false);
                var offlineTenantPrefixOk = useMock &&
                    submittedKey.StartsWith($"ares_tnt_{orgId}_", StringComparison.OrdinalIgnoreCase);
                if (!offlineTenantPrefixOk)
                {
                    return new AresAuthResult(
                        false,
                        "Could not verify tenant API key against backend and no matching configured tenant key was found."
                    );
                }
            }
        }

        return IssueSession(email, request.DisplayName, "Tenant", orgId, submittedKey);
    }

    private AresAuthResult IssueSession(
        string email,
        string? requestedDisplayName,
        string role,
        string? orgId,
        string apiKey)
    {
        var displayName = string.IsNullOrWhiteSpace(requestedDisplayName)
            ? email.Split('@')[0]
            : requestedDisplayName.Trim();
        var initials = AresSessionRegistry.ComputeInitials(displayName, email);
        var sid = Convert.ToHexString(RandomNumberGenerator.GetBytes(24)).ToLowerInvariant();

        var session = new AresOperatorSession(
            SessionId: sid,
            Email: email,
            DisplayName: displayName,
            Initials: initials,
            Role: role,
            OrganizationId: orgId,
            EffectiveApiKey: apiKey,
            AuthenticatedAt: DateTimeOffset.UtcNow
        );
        _registry.Store(session);

        var claims = new List<Claim>
        {
            new(ClaimTypes.Name, displayName),
            new(ClaimTypes.Email, email),
            new(ClaimTypes.Role, role),
            new(AresOperatorContext.SessionIdClaim, sid),
            new(AresOperatorContext.InitialsClaim, initials),
            new(AresOperatorContext.OrgIdClaim, orgId ?? "*"),
        };
        var identity = new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme);
        var principal = new ClaimsPrincipal(identity);

        return new AresAuthResult(true, null, session, principal);
    }

    private Dictionary<string, string> ParseConfiguredTenantKeys()
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        var rawEnv = _config["ARES_TENANT_API_KEYS"] ?? Environment.GetEnvironmentVariable("ARES_TENANT_API_KEYS");
        if (!string.IsNullOrWhiteSpace(rawEnv))
        {
            var trimmed = rawEnv.Trim();
            if (trimmed.StartsWith('{'))
            {
                try
                {
                    var map = JsonSerializer.Deserialize<Dictionary<string, string>>(trimmed);
                    if (map is not null)
                    {
                        foreach (var kv in map)
                        {
                            if (!string.IsNullOrWhiteSpace(kv.Key) && !string.IsNullOrWhiteSpace(kv.Value))
                                result[kv.Key.Trim()] = kv.Value.Trim();
                        }
                    }
                }
                catch { /* ignore malformed env json */ }
            }
            else
            {
                foreach (var entry in trimmed.Split(',', StringSplitOptions.RemoveEmptyEntries))
                {
                    var idx = entry.IndexOf(':');
                    if (idx > 0 && idx < entry.Length - 1)
                    {
                        var k = entry[..idx].Trim();
                        var v = entry[(idx + 1)..].Trim();
                        if (!string.IsNullOrWhiteSpace(k) && !string.IsNullOrWhiteSpace(v))
                            result[k] = v;
                    }
                }
            }
        }

        // Explicit IConfiguration section entries take precedence over ambient env defaults.
        var section = _config.GetSection("Ares:TenantApiKeys");
        foreach (var child in section.GetChildren())
        {
            if (!string.IsNullOrWhiteSpace(child.Key) && !string.IsNullOrWhiteSpace(child.Value))
            {
                result[child.Key.Trim()] = child.Value.Trim();
            }
        }

        return result;
    }

    private async Task<bool> VerifyKeyAgainstBackendAsync(
        string apiKey,
        string? orgId,
        bool expectAdmin,
        CancellationToken cancellationToken)
    {
        try
        {
            using var msg = new HttpRequestMessage(HttpMethod.Get, "api/v1/auth/whoami");
            msg.Headers.TryAddWithoutValidation("X-Ares-Api-Key", apiKey);
            using var resp = await _http.SendAsync(msg, cancellationToken);
            if (!resp.IsSuccessStatusCode) return false;

            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(cancellationToken));
            var root = doc.RootElement;
            var isAuthenticated = root.TryGetProperty("authenticated", out var authProp) && authProp.ValueKind == JsonValueKind.True;
            if (!isAuthenticated)
            {
                return false;
            }
            var isAdmin = root.TryGetProperty("is_admin", out var adminProp) && adminProp.ValueKind == JsonValueKind.True;
            var scopedOrg = root.TryGetProperty("organization_id", out var orgProp) && orgProp.ValueKind == JsonValueKind.String
                ? orgProp.GetString()
                : null;

            if (expectAdmin)
            {
                return isAdmin && string.IsNullOrWhiteSpace(scopedOrg);
            }

            if (isAdmin) return true;
            return !string.IsNullOrWhiteSpace(orgId) &&
                   string.Equals(scopedOrg, orgId, StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private async Task<(bool ReachedBackend, bool Authorized)> VerifyTenantKeyWithStatusAsync(
        string apiKey,
        string orgId,
        CancellationToken cancellationToken)
    {
        try
        {
            using var msg = new HttpRequestMessage(HttpMethod.Get, "api/v1/auth/whoami");
            msg.Headers.TryAddWithoutValidation("X-Ares-Api-Key", apiKey);
            msg.Headers.TryAddWithoutValidation("X-Ares-Organization-Id", orgId);
            using var resp = await _http.SendAsync(msg, cancellationToken);
            if (resp.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
            {
                return (true, false);
            }
            if (!resp.IsSuccessStatusCode)
            {
                return (true, false);
            }
            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(cancellationToken));
            var root = doc.RootElement;
            var isAuthenticated = root.TryGetProperty("authenticated", out var authProp) && authProp.ValueKind == JsonValueKind.True;
            if (!isAuthenticated)
            {
                return (true, false);
            }
            var isAdmin = root.TryGetProperty("is_admin", out var adminProp) && adminProp.ValueKind == JsonValueKind.True;
            var scopedOrg = root.TryGetProperty("organization_id", out var orgProp) && orgProp.ValueKind == JsonValueKind.String
                ? orgProp.GetString()
                : null;
            var authorized = isAdmin || string.Equals(scopedOrg, orgId, StringComparison.OrdinalIgnoreCase);
            return (true, authorized);
        }
        catch
        {
            return (false, false);
        }
    }

    public static bool ConstantTimeEquals(string left, string right)
    {
        var leftBytes = Encoding.UTF8.GetBytes(left ?? string.Empty);
        var rightBytes = Encoding.UTF8.GetBytes(right ?? string.Empty);
        if (leftBytes.Length != rightBytes.Length) return false;
        return CryptographicOperations.FixedTimeEquals(leftBytes, rightBytes);
    }

    public static bool IsSafeLocalReturnUrl(string? returnUrl)
    {
        if (string.IsNullOrWhiteSpace(returnUrl)) return false;
        if (!returnUrl.StartsWith('/')) return false;
        if (returnUrl.Length > 1 && (returnUrl[1] == '/' || returnUrl[1] == '\\')) return false;
        return !returnUrl.Contains('\r') && !returnUrl.Contains('\n');
    }
}
