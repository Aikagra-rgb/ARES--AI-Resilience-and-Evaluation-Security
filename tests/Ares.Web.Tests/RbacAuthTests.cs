using System.Net;
using System.Security.Claims;
using System.Text;
using Ares.Web.Services;
using Microsoft.AspNetCore.Antiforgery;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Components.Authorization;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Xunit;

namespace Ares.Web.Tests;

public sealed class RbacAuthTests
{
    private static IConfiguration BuildConfig(Dictionary<string, string?>? overrides = null)
    {
        var defaults = new Dictionary<string, string?>
        {
            ["Ares:ApiBaseUrl"] = "http://localhost:8000",
            ["Ares:ApiKey"] = "ares-master-admin-secret-key",
            ["Ares:TenantApiKeys:org_acme_fintech"] = "ares_tnt_org_acme_fintech_key01",
            ["Ares:TenantApiKeys:org_medshield_health"] = "ares_tnt_org_medshield_health_key02",
        };
        if (overrides is not null)
        {
            foreach (var kv in overrides)
                defaults[kv.Key] = kv.Value;
        }
        return new ConfigurationBuilder().AddInMemoryCollection(defaults).Build();
    }

    private sealed class StubHttpMessageHandler(Func<HttpRequestMessage, HttpResponseMessage> responder) : HttpMessageHandler
    {
        public HttpRequestMessage? LastRequest { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            LastRequest = request;
            return Task.FromResult(responder(request));
        }
    }

    [Fact]
    public async Task Admin_authentication_succeeds_with_master_key_and_computes_initials()
    {
        var registry = new AresSessionRegistry();
        var auth = new AresAuthService(BuildConfig(), registry);

        var result = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "alice.vance@ares.security",
            DisplayName: "Alice Vance",
            Role: "Admin",
            OrganizationId: null,
            ApiKey: "ares-master-admin-secret-key"
        ));

        Assert.True(result.Succeeded);
        Assert.NotNull(result.Session);
        Assert.NotNull(result.Principal);
        Assert.True(result.Session!.IsAdmin);
        Assert.Equal("Admin", result.Session.Role);
        Assert.Null(result.Session.OrganizationId);
        Assert.Equal("AV", result.Session.Initials);
        Assert.True(result.Session.CanAccessOrganization("org_acme_fintech"));
        Assert.True(result.Session.CanAccessOrganization("org_medshield_health"));
        Assert.Equal("Admin", result.Principal!.FindFirst(ClaimTypes.Role)?.Value);
    }

    [Fact]
    public async Task Tenant_key_cannot_authenticate_as_admin()
    {
        var registry = new AresSessionRegistry();
        var auth = new AresAuthService(BuildConfig(), registry);

        var result = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "tenant@acme.test",
            DisplayName: "Acme Analyst",
            Role: "Admin",
            OrganizationId: null,
            ApiKey: "ares_tnt_org_acme_fintech_key01"
        ));

        Assert.False(result.Succeeded);
        Assert.Contains("Tenant-scoped", result.ErrorMessage ?? string.Empty);
    }

    [Fact]
    public async Task Tenant_authentication_scopes_operator_to_single_org_and_blocks_cross_tenant_key()
    {
        var registry = new AresSessionRegistry();
        var auth = new AresAuthService(BuildConfig(), registry);

        // Valid tenant key for own org
        var ok = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "analyst@acme.test",
            DisplayName: "Jordan Patel",
            Role: "Tenant",
            OrganizationId: "org_acme_fintech",
            ApiKey: "ares_tnt_org_acme_fintech_key01"
        ));
        Assert.True(ok.Succeeded);
        Assert.NotNull(ok.Session);
        Assert.False(ok.Session!.IsAdmin);
        Assert.Equal("org_acme_fintech", ok.Session.OrganizationId);
        Assert.Equal("JP", ok.Session.Initials);
        Assert.True(ok.Session.CanAccessOrganization("org_acme_fintech"));
        Assert.False(ok.Session.CanAccessOrganization("org_medshield_health"));

        // Cross-tenant key for medshield attempting to sign in to acme_fintech -> rejected
        var denied = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "analyst@acme.test",
            DisplayName: "Jordan Patel",
            Role: "Tenant",
            OrganizationId: "org_acme_fintech",
            ApiKey: "ares_tnt_org_medshield_health_key02"
        ));
        Assert.False(denied.Succeeded);
        Assert.Contains("different organization", denied.ErrorMessage ?? string.Empty);

        // Missing org_id -> rejected
        var missingOrg = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "analyst@acme.test",
            DisplayName: "Jordan Patel",
            Role: "Tenant",
            OrganizationId: "",
            ApiKey: "ares_tnt_org_acme_fintech_key01"
        ));
        Assert.False(missingOrg.Succeeded);
    }

    [Fact]
    public async Task Operator_context_applies_tenant_scoped_headers_overriding_global_key()
    {
        var registry = new AresSessionRegistry();
        var auth = new AresAuthService(BuildConfig(), registry);
        var login = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "secops@acme.test",
            DisplayName: "Maya Lin",
            Role: "Tenant",
            OrganizationId: "org_acme_fintech",
            ApiKey: "ares_tnt_org_acme_fintech_key01"
        ));
        Assert.True(login.Succeeded);

        var opCtx = new AresOperatorContext(registry);
        opCtx.SetSession(login.Session);

        var handler = new StubHttpMessageHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent("{\"status\":\"ACTIVE\",\"qdrant_points\":100,\"block_threshold\":0.8,\"sanitize_threshold\":0.65,\"canary_token_configured\":true,\"version\":\"1.0.0\"}", Encoding.UTF8, "application/json")
        });
        using var http = new HttpClient(handler) { BaseAddress = new Uri("http://localhost:8000/") };
        http.DefaultRequestHeaders.Add("X-Ares-Api-Key", "global-should-be-overridden");

        var apiClient = new FastApiAresApiClient(http, new MockAresApiClient(), operatorContext: opCtx);
        var status = await apiClient.GetGatewayStatusAsync(CancellationToken.None);

        Assert.Equal("ACTIVE", status.Status);
        Assert.NotNull(handler.LastRequest);
        Assert.Equal("ares_tnt_org_acme_fintech_key01", handler.LastRequest!.Headers.GetValues("X-Ares-Api-Key").Single());
        Assert.Equal("org_acme_fintech", handler.LastRequest.Headers.GetValues("X-Ares-Organization-Id").Single());
        Assert.Equal("secops@acme.test", handler.LastRequest.Headers.GetValues("X-Ares-User-Email").Single());
    }

    [Fact]
    public async Task Session_removal_revokes_principal_resolution()
    {
        var registry = new AresSessionRegistry();
        var auth = new AresAuthService(BuildConfig(), registry);
        var login = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "ops@ares.local",
            DisplayName: "Ops Admin",
            Role: "Admin",
            OrganizationId: null,
            ApiKey: "ares-master-admin-secret-key"
        ));
        Assert.True(login.Succeeded);

        var opCtx = new AresOperatorContext(registry);
        Assert.NotNull(opCtx.ResolveFromPrincipal(login.Principal));

        // Revoking session on sign-out invalidates cookie principal lookup
        Assert.True(registry.Remove(login.Session!.SessionId));
        Assert.Null(opCtx.ResolveFromPrincipal(login.Principal));
    }

    [Fact]
    public async Task Backend_whoami_blocks_non_prefixed_tenant_key_from_admin_escalation()
    {
        var registry = new AresSessionRegistry();
        var handler = new StubHttpMessageHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(
                "{\"authenticated\":true,\"role\":\"tenant\",\"is_admin\":false,\"organization_id\":\"org_acme_fintech\"}",
                Encoding.UTF8,
                "application/json"
            )
        });
        using var http = new HttpClient(handler) { BaseAddress = new Uri("http://localhost:8000/") };
        var auth = new AresAuthService(BuildConfig(), registry, httpClient: http);

        // Non-prefixed custom env tenant key attempting Admin login -> must be rejected by whoami
        var adminAttempt = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "attacker@acme.test",
            DisplayName: "Sneaky Tenant",
            Role: "Admin",
            OrganizationId: null,
            ApiKey: "custom-env-tenant-secret-without-prefix"
        ));
        Assert.False(adminAttempt.Succeeded);

        // Same key logging in as Tenant for its actual org -> succeeds
        var tenantAttempt = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "analyst@acme.test",
            DisplayName: "Acme Analyst",
            Role: "Tenant",
            OrganizationId: "org_acme_fintech",
            ApiKey: "custom-env-tenant-secret-without-prefix"
        ));
        Assert.True(tenantAttempt.Succeeded);
        Assert.Equal("org_acme_fintech", tenantAttempt.Session!.OrganizationId);

        // Same key attempting Tenant login for a DIFFERENT org -> rejected by whoami
        var crossOrgAttempt = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "analyst@acme.test",
            DisplayName: "Acme Analyst",
            Role: "Tenant",
            OrganizationId: "org_medshield_health",
            ApiKey: "custom-env-tenant-secret-without-prefix"
        ));
        Assert.False(crossOrgAttempt.Succeeded);
    }

    [Fact]
    public async Task Backend_whoami_unauthenticated_anonymous_response_cannot_authenticate()
    {
        var registry = new AresSessionRegistry();
        // Simulates backend with ARES_AUTH_ENABLED=false returning authenticated=false, is_admin=true
        var handler = new StubHttpMessageHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(
                "{\"authenticated\":false,\"role\":\"anonymous\",\"is_admin\":true,\"organization_id\":null}",
                Encoding.UTF8,
                "application/json"
            )
        });
        using var http = new HttpClient(handler) { BaseAddress = new Uri("http://localhost:8000/") };
        var auth = new AresAuthService(BuildConfig(), registry, httpClient: http);

        var adminAttempt = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "intruder@evil.test",
            DisplayName: "Intruder",
            Role: "Admin",
            OrganizationId: null,
            ApiKey: "completely-random-invalid-key"
        ));
        Assert.False(adminAttempt.Succeeded);

        var tenantAttempt = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "intruder@evil.test",
            DisplayName: "Intruder",
            Role: "Tenant",
            OrganizationId: "org_acme_fintech",
            ApiKey: "completely-random-invalid-key"
        ));
        Assert.False(tenantAttempt.Succeeded);
    }

    private sealed class StaticAuthStateProvider(ClaimsPrincipal principal) : AuthenticationStateProvider
    {
        public override Task<AuthenticationState> GetAuthenticationStateAsync()
            => Task.FromResult(new AuthenticationState(principal));
    }

    [Fact]
    public async Task Operator_context_fails_closed_without_session_and_hydrates_async_via_auth_state_provider()
    {
        var registry = new AresSessionRegistry();
        var auth = new AresAuthService(BuildConfig(), registry);
        var login = await auth.AuthenticateAsync(new AresLoginRequest(
            Email: "circuit@acme.test",
            DisplayName: "Circuit User",
            Role: "Tenant",
            OrganizationId: "org_acme_fintech",
            ApiKey: "ares_tnt_org_acme_fintech_key01"
        ));
        Assert.True(login.Succeeded);

        // 1. Hydrates asynchronously via AuthenticationStateProvider (no manual SetSession, no HttpContext)
        var opCtx = new AresOperatorContext(registry, authStateProvider: new StaticAuthStateProvider(login.Principal!));
        var handler = new StubHttpMessageHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent("{\"status\":\"ACTIVE\",\"qdrant_points\":10,\"block_threshold\":0.8,\"sanitize_threshold\":0.65,\"canary_token_configured\":true,\"version\":\"1.0.0\"}", Encoding.UTF8, "application/json")
        });
        using var http = new HttpClient(handler) { BaseAddress = new Uri("http://localhost:8000/") };
        var apiClient = new FastApiAresApiClient(http, new MockAresApiClient(), operatorContext: opCtx);
        var status = await apiClient.GetGatewayStatusAsync(CancellationToken.None);
        Assert.Equal("ACTIVE", status.Status);
        Assert.Equal("ares_tnt_org_acme_fintech_key01", handler.LastRequest!.Headers.GetValues("X-Ares-Api-Key").Single());
        Assert.Equal("org_acme_fintech", handler.LastRequest.Headers.GetValues("X-Ares-Organization-Id").Single());

        // 2. Evicted/missing session MUST fail closed (throw UnauthorizedAccessException) rather than leaking any ambient key
        var emptyCtx = new AresOperatorContext(registry);
        var evictedClient = new FastApiAresApiClient(http, new MockAresApiClient(), operatorContext: emptyCtx);
        await Assert.ThrowsAsync<UnauthorizedAccessException>(() => evictedClient.GetGatewayStatusAsync(CancellationToken.None));

        // 3. Tenant self-key-rotation updates session EffectiveApiKey in place
        var rotated = registry.UpdateApiKey(login.Session!.SessionId, "ares_tnt_org_acme_fintech_rotated99");
        Assert.NotNull(rotated);
        Assert.Equal("ares_tnt_org_acme_fintech_rotated99", rotated!.EffectiveApiKey);
    }

    [Fact]
    public void Return_url_validator_blocks_open_redirects_including_backslash_bypass()
    {
        Assert.True(AresAuthService.IsSafeLocalReturnUrl("/"));
        Assert.True(AresAuthService.IsSafeLocalReturnUrl("/clients?tab=models"));
        Assert.False(AresAuthService.IsSafeLocalReturnUrl(""));
        Assert.False(AresAuthService.IsSafeLocalReturnUrl("https://evil.example/phish"));
        Assert.False(AresAuthService.IsSafeLocalReturnUrl("//evil.example"));
        Assert.False(AresAuthService.IsSafeLocalReturnUrl("/\\evil.example"));
        Assert.False(AresAuthService.IsSafeLocalReturnUrl("/clients\r\nSet-Cookie: bad=1"));
    }

    [Fact]
    public async Task Cookie_auth_http_roundtrip_issues_httponly_cookie_and_rejects_evicted_sessions()
    {
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Configuration.AddInMemoryCollection(new Dictionary<string, string?>
        {
            ["Ares:ApiBaseUrl"] = "http://localhost:8000",
            ["Ares:ApiKey"] = "ares-master-admin-secret-key",
            ["Ares:TenantApiKeys:org_acme_fintech"] = "ares_tnt_org_acme_fintech_key01",
        });
        builder.Services.AddSingleton<IDataProtectionProvider, EphemeralDataProtectionProvider>();
        builder.Services.AddSingleton<AresSessionRegistry>();
        builder.Services.AddSingleton<AresAuthService>();
        builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
            .AddCookie(options =>
            {
                options.Cookie.Name = "ares.console.auth";
                options.Cookie.HttpOnly = true;
                options.Cookie.SameSite = SameSiteMode.Lax;
                options.LoginPath = "/login";
                options.Events = new CookieAuthenticationEvents
                {
                    OnValidatePrincipal = async context =>
                    {
                        var sid = context.Principal?.FindFirst(AresOperatorContext.SessionIdClaim)?.Value;
                        var reg = context.HttpContext.RequestServices.GetRequiredService<AresSessionRegistry>();
                        if (!reg.TryGet(sid, out _))
                        {
                            context.RejectPrincipal();
                            await context.HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
                        }
                    }
                };
            });
        builder.Services.AddAuthorization();

        await using var app = builder.Build();
        app.UseAuthentication();
        app.UseAuthorization();

        app.MapPost("/auth/login", async (HttpContext ctx, AresAuthService authSvc) =>
        {
            var form = await ctx.Request.ReadFormAsync();
            var res = await authSvc.AuthenticateAsync(new AresLoginRequest(
                form["email"].ToString(),
                form["display_name"].ToString(),
                form["role"].ToString(),
                form["org_id"].ToString(),
                form["api_key"].ToString()
            ));
            if (!res.Succeeded || res.Principal is null)
                return Results.Redirect("/login?error=denied");
            await ctx.SignInAsync(CookieAuthenticationDefaults.AuthenticationScheme, res.Principal);
            return Results.Redirect("/");
        }).AllowAnonymous();

        app.MapPost("/auth/logout", async (HttpContext ctx, AresSessionRegistry reg) =>
        {
            var sid = ctx.User.FindFirst(AresOperatorContext.SessionIdClaim)?.Value;
            reg.Remove(sid);
            await ctx.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
            return Results.Redirect("/login");
        }).AllowAnonymous();

        app.MapGet("/protected", (HttpContext ctx) =>
            Results.Ok(new
            {
                role = ctx.User.FindFirst(ClaimTypes.Role)?.Value,
                org = ctx.User.FindFirst(AresOperatorContext.OrgIdClaim)?.Value,
                sid = ctx.User.FindFirst(AresOperatorContext.SessionIdClaim)?.Value,
            })
        ).RequireAuthorization();

        await app.StartAsync();
        try
        {
            var address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
            var cookies = new CookieContainer();
            using var handler = new HttpClientHandler { CookieContainer = cookies, AllowAutoRedirect = false };
            using var client = new HttpClient(handler) { BaseAddress = new Uri(address) };

            // 1. Unauthenticated request redirects to /login
            var unauth = await client.GetAsync("/protected");
            Assert.Equal(HttpStatusCode.Redirect, unauth.StatusCode);
            Assert.Contains("/login", unauth.Headers.Location?.ToString() ?? "");

            // 2. POST /auth/login issues HttpOnly cookie
            var loginResp = await client.PostAsync("/auth/login", new FormUrlEncodedContent(new Dictionary<string, string>
            {
                ["email"] = "analyst@acme.test",
                ["display_name"] = "Jordan Patel",
                ["role"] = "Tenant",
                ["org_id"] = "org_acme_fintech",
                ["api_key"] = "ares_tnt_org_acme_fintech_key01",
            }));
            Assert.Equal(HttpStatusCode.Redirect, loginResp.StatusCode);
            var setCookieHeader = string.Join("; ", loginResp.Headers.GetValues("Set-Cookie"));
            Assert.Contains("ares.console.auth=", setCookieHeader);
            Assert.Contains("httponly", setCookieHeader, StringComparison.OrdinalIgnoreCase);

            // 3. Authenticated request with cookie succeeds
            var authed = await client.GetAsync("/protected");
            Assert.Equal(HttpStatusCode.OK, authed.StatusCode);
            var authedBody = await authed.Content.ReadAsStringAsync();
            Assert.Contains("org_acme_fintech", authedBody);

            // 4. Simulate server restart / session eviction -> OnValidatePrincipal rejects stale cookie and redirects to /login
            var reg = app.Services.GetRequiredService<AresSessionRegistry>();
            using var doc = System.Text.Json.JsonDocument.Parse(authedBody);
            var sid = doc.RootElement.GetProperty("sid").GetString();
            Assert.True(reg.Remove(sid));

            var afterEviction = await client.GetAsync("/protected");
            Assert.Equal(HttpStatusCode.Redirect, afterEviction.StatusCode);
            Assert.Contains("/login", afterEviction.Headers.Location?.ToString() ?? "");

            // 5. Re-authenticate, then POST /auth/logout clears session & cookie
            var loginAgain = await client.PostAsync("/auth/login", new FormUrlEncodedContent(new Dictionary<string, string>
            {
                ["email"] = "analyst@acme.test",
                ["display_name"] = "Jordan Patel",
                ["role"] = "Tenant",
                ["org_id"] = "org_acme_fintech",
                ["api_key"] = "ares_tnt_org_acme_fintech_key01",
            }));
            Assert.Equal(HttpStatusCode.Redirect, loginAgain.StatusCode);
            Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/protected")).StatusCode);

            var logoutResp = await client.PostAsync("/auth/logout", content: null);
            Assert.Equal(HttpStatusCode.Redirect, logoutResp.StatusCode);
            Assert.Contains("/login", logoutResp.Headers.Location?.ToString() ?? "");

            var afterLogout = await client.GetAsync("/protected");
            Assert.Equal(HttpStatusCode.Redirect, afterLogout.StatusCode);
        }
        finally
        {
            await app.StopAsync();
        }
    }

    [Fact]
    public void Session_registry_persists_encrypted_sessions_across_restarts_without_plaintext_secrets()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"ares-session-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var keysDir = new DirectoryInfo(Path.Combine(tempDir, "keys"));
            var sessionPath = Path.Combine(tempDir, "operator-sessions.protected.json");
            var sp1 = new ServiceCollection()
                .AddDataProtection()
                .PersistKeysToFileSystem(keysDir)
                .SetApplicationName("ARES-Test")
                .Services.BuildServiceProvider();

            var registry1 = new AresSessionRegistry(
                dataProtectionProvider: sp1.GetRequiredService<IDataProtectionProvider>(),
                persistencePath: sessionPath
            );
            const string secretKey = "ares_tnt_org_acme_fintech_supersecret_991";
            var session = new AresOperatorSession(
                SessionId: "sid-persist-001",
                Email: "secops@acme.test",
                DisplayName: "Sec Ops",
                Initials: "SO",
                Role: "Tenant",
                OrganizationId: "org_acme_fintech",
                EffectiveApiKey: secretKey,
                AuthenticatedAt: DateTimeOffset.UtcNow
            );
            registry1.Store(session);

            Assert.True(File.Exists(sessionPath));
            var rawDiskContent = File.ReadAllText(sessionPath);
            Assert.DoesNotContain(secretKey, rawDiskContent);
            Assert.DoesNotContain("secops@acme.test", rawDiskContent);

            // Simulate process restart with new ServiceProvider + registry reading the same DataProtection keys
            var sp2 = new ServiceCollection()
                .AddDataProtection()
                .PersistKeysToFileSystem(keysDir)
                .SetApplicationName("ARES-Test")
                .Services.BuildServiceProvider();
            var registry2 = new AresSessionRegistry(
                dataProtectionProvider: sp2.GetRequiredService<IDataProtectionProvider>(),
                persistencePath: sessionPath
            );

            Assert.True(registry2.TryGet("sid-persist-001", out var restored));
            Assert.NotNull(restored);
            Assert.Equal(secretKey, restored!.EffectiveApiKey);
            Assert.Equal("org_acme_fintech", restored.OrganizationId);

            // Eviction persists to disk too
            Assert.True(registry2.Remove("sid-persist-001"));
            var registry3 = new AresSessionRegistry(
                dataProtectionProvider: sp2.GetRequiredService<IDataProtectionProvider>(),
                persistencePath: sessionPath
            );
            Assert.False(registry3.TryGet("sid-persist-001", out _));
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, recursive: true);
            }
        }
    }

    [Fact]
    public void Session_registry_supports_encrypted_distributed_cache_fallback()
    {
        var dp = DataProtectionProvider.Create("ARES-Cache-Test");
        var cache = new MemoryDistributedCache(Options.Create(new MemoryDistributedCacheOptions()));

        var writerRegistry = new AresSessionRegistry(dp, persistencePath: null, distributedCache: cache);
        var session = new AresOperatorSession(
            SessionId: "sid-cache-77",
            Email: "admin@ares.local",
            DisplayName: "Admin",
            Initials: "AD",
            Role: "Admin",
            OrganizationId: null,
            EffectiveApiKey: "ares-distributed-secret-xyz",
            AuthenticatedAt: DateTimeOffset.UtcNow
        );
        writerRegistry.Store(session);

        // Fresh replica sharing distributed cache hydrates the encrypted session
        var replicaRegistry = new AresSessionRegistry(dp, persistencePath: null, distributedCache: cache);
        Assert.True(replicaRegistry.TryGet("sid-cache-77", out var restored));
        Assert.Equal("ares-distributed-secret-xyz", restored!.EffectiveApiKey);

        Assert.True(replicaRegistry.Remove("sid-cache-77"));
        Assert.False(writerRegistry.TryGet("sid-cache-77", out _) && replicaRegistry.TryGet("sid-cache-77", out _));
    }

    [Fact]
    public async Task Auth_endpoints_enforce_csrf_antiforgery_and_protect_get_logout()
    {
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Configuration.AddInMemoryCollection(new Dictionary<string, string?>
        {
            ["Ares:ApiBaseUrl"] = "http://localhost:8000",
            ["Ares:ApiKey"] = "ares-master-admin-secret-key",
            ["Ares:TenantApiKeys:org_acme_fintech"] = "ares_tnt_org_acme_fintech_key01",
        });
        builder.Services.AddSingleton<IDataProtectionProvider, EphemeralDataProtectionProvider>();
        builder.Services.AddAntiforgery();
        builder.Services.AddSingleton<AresSessionRegistry>(sp =>
            new AresSessionRegistry(sp.GetRequiredService<IDataProtectionProvider>()));
        builder.Services.AddSingleton<AresAuthService>();
        builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
            .AddCookie(options =>
            {
                options.Cookie.Name = "ares.console.auth";
                options.Cookie.HttpOnly = true;
                options.Cookie.SameSite = SameSiteMode.Lax;
                options.LoginPath = "/login";
            });
        builder.Services.AddAuthorization();

        await using var app = builder.Build();
        app.UseAuthentication();
        app.UseAuthorization();
        app.UseAntiforgery();

        app.MapGet("/csrf-token", (HttpContext ctx, IAntiforgery antiforgery) =>
        {
            var tokens = antiforgery.GetAndStoreTokens(ctx);
            return Results.Ok(new { field = tokens.FormFieldName, token = tokens.RequestToken });
        }).AllowAnonymous();

        app.MapPost("/auth/login", async (HttpContext ctx, AresAuthService authSvc, IAntiforgery antiforgery) =>
        {
            try
            {
                await antiforgery.ValidateRequestAsync(ctx);
            }
            catch (AntiforgeryValidationException)
            {
                return Results.BadRequest(new { error = "invalid_antiforgery_token" });
            }
            var form = await ctx.Request.ReadFormAsync();
            var res = await authSvc.AuthenticateAsync(new AresLoginRequest(
                form["email"].ToString(),
                form["display_name"].ToString(),
                form["role"].ToString(),
                form["org_id"].ToString(),
                form["api_key"].ToString()
            ));
            if (!res.Succeeded || res.Principal is null)
                return Results.Redirect("/login?error=denied");
            await ctx.SignInAsync(CookieAuthenticationDefaults.AuthenticationScheme, res.Principal);
            return Results.Redirect("/");
        }).AllowAnonymous();

        app.MapGet("/auth/logout", async (HttpContext ctx, AresSessionRegistry reg) =>
        {
            var sid = ctx.User.FindFirst(AresOperatorContext.SessionIdClaim)?.Value;
            if (ctx.User.Identity?.IsAuthenticated == true && reg.TryGet(sid, out _))
            {
                return Results.Redirect("/");
            }
            await ctx.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
            return Results.Redirect("/login");
        }).AllowAnonymous();

        app.MapPost("/auth/logout", async (HttpContext ctx, AresSessionRegistry reg, IAntiforgery antiforgery) =>
        {
            try
            {
                await antiforgery.ValidateRequestAsync(ctx);
            }
            catch (AntiforgeryValidationException)
            {
                return Results.BadRequest(new { error = "invalid_antiforgery_token" });
            }
            var sid = ctx.User.FindFirst(AresOperatorContext.SessionIdClaim)?.Value;
            reg.Remove(sid);
            await ctx.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
            return Results.Redirect("/login");
        }).AllowAnonymous();

        app.MapGet("/protected", (HttpContext ctx) =>
            Results.Ok(new { sid = ctx.User.FindFirst(AresOperatorContext.SessionIdClaim)?.Value })
        ).RequireAuthorization();

        await app.StartAsync();
        try
        {
            var address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
            var cookies = new CookieContainer();
            using var handler = new HttpClientHandler { CookieContainer = cookies, AllowAutoRedirect = false };
            using var client = new HttpClient(handler) { BaseAddress = new Uri(address) };

            // 1. POST /auth/login without CSRF token MUST fail with 400
            var missingCsrfLogin = await client.PostAsync("/auth/login", new FormUrlEncodedContent(new Dictionary<string, string>
            {
                ["email"] = "analyst@acme.test",
                ["role"] = "Tenant",
                ["org_id"] = "org_acme_fintech",
                ["api_key"] = "ares_tnt_org_acme_fintech_key01",
            }));
            Assert.Equal(HttpStatusCode.BadRequest, missingCsrfLogin.StatusCode);

            // 2. Fetch CSRF token + cookie and sign in cleanly
            var tokenDoc = System.Text.Json.JsonDocument.Parse(await client.GetStringAsync("/csrf-token"));
            var afField = tokenDoc.RootElement.GetProperty("field").GetString()!;
            var afToken = tokenDoc.RootElement.GetProperty("token").GetString()!;

            var validLogin = await client.PostAsync("/auth/login", new FormUrlEncodedContent(new Dictionary<string, string>
            {
                [afField] = afToken,
                ["email"] = "analyst@acme.test",
                ["role"] = "Tenant",
                ["org_id"] = "org_acme_fintech",
                ["api_key"] = "ares_tnt_org_acme_fintech_key01",
            }));
            Assert.Equal(HttpStatusCode.Redirect, validLogin.StatusCode);
            Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/protected")).StatusCode);

            // 3. GET /auth/logout must NOT log out an active session (redirects to /)
            var getLogoutActive = await client.GetAsync("/auth/logout");
            Assert.Equal(HttpStatusCode.Redirect, getLogoutActive.StatusCode);
            Assert.Equal("/", getLogoutActive.Headers.Location?.ToString());
            Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/protected")).StatusCode);

            // 4. POST /auth/logout without CSRF token MUST fail with 400 and leave session active
            var missingCsrfLogout = await client.PostAsync("/auth/logout", content: null);
            Assert.Equal(HttpStatusCode.BadRequest, missingCsrfLogout.StatusCode);
            Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/protected")).StatusCode);

            // 5. Fetch authenticated CSRF token and perform valid POST /auth/logout
            var authTokenDoc = System.Text.Json.JsonDocument.Parse(await client.GetStringAsync("/csrf-token"));
            var validLogout = await client.PostAsync("/auth/logout", new FormUrlEncodedContent(new Dictionary<string, string>
            {
                [authTokenDoc.RootElement.GetProperty("field").GetString()!] = authTokenDoc.RootElement.GetProperty("token").GetString()!,
            }));
            Assert.Equal(HttpStatusCode.Redirect, validLogout.StatusCode);
            Assert.Contains("/login", validLogout.Headers.Location?.ToString() ?? "");
            Assert.Equal(HttpStatusCode.Redirect, (await client.GetAsync("/protected")).StatusCode);
        }
        finally
        {
            await app.StopAsync();
        }
    }
}
