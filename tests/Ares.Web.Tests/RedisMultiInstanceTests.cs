using System.Net;
using Ares.Web.Services;
using Microsoft.AspNetCore.Antiforgery;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using StackExchange.Redis;
using Xunit;

namespace Ares.Web.Tests;

public sealed class RedisMultiInstanceTests
{
    [Fact]
    public async Task Live_redis_shares_dataprotection_keys_and_revokes_sessions_across_two_instances()
    {
        var rawRedis = Environment.GetEnvironmentVariable("ARES_LIVE_REDIS_CONNECTION");
        var requireLive = string.Equals(
            Environment.GetEnvironmentVariable("ARES_REQUIRE_LIVE_REDIS"),
            "1",
            StringComparison.OrdinalIgnoreCase
        );

        if (string.IsNullOrWhiteSpace(rawRedis))
        {
            if (requireLive)
            {
                Assert.Fail("ARES_REQUIRE_LIVE_REDIS=1 is set but ARES_LIVE_REDIS_CONNECTION is empty.");
            }
            return;
        }

        var normalized = AresRedisConfiguration.NormalizeConnectionString(rawRedis);
        using var mux = await ConnectionMultiplexer.ConnectAsync(normalized);
        var keyPrefix = $"ares:test:{Guid.NewGuid():N}:";

        async Task<(WebApplication app, string baseUrl)> StartRedisBackedNodeAsync()
        {
            var builder = WebApplication.CreateBuilder();
            builder.WebHost.UseUrls("http://127.0.0.1:0");
            builder.Configuration.AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["Ares:ApiBaseUrl"] = "http://localhost:8000",
                ["Ares:ApiKey"] = "ares-master-admin-secret-key",
                ["Ares:TenantApiKeys:org_acme_fintech"] = "ares_tnt_org_acme_fintech_key01",
                ["ARES_TENANT_API_KEYS"] = "",
            });
            builder.Services.AddStackExchangeRedisCache(o =>
            {
                o.ConnectionMultiplexerFactory = () => Task.FromResult<IConnectionMultiplexer>(mux);
                o.InstanceName = keyPrefix;
            });
            builder.Services.AddDataProtection()
                .SetApplicationName("ARES-LiveRedis-Test")
                .PersistKeysToStackExchangeRedis(mux, $"{keyPrefix}dpkeys");
            builder.Services.AddAntiforgery();
            builder.Services.AddSingleton<AresSessionRegistry>(sp => new AresSessionRegistry(
                sp.GetRequiredService<IDataProtectionProvider>(),
                persistencePath: null,
                distributedCache: sp.GetRequiredService<IDistributedCache>(),
                sessionLifetime: TimeSpan.FromHours(1)
            ));
            builder.Services.AddSingleton<AresAuthService>();
            builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
                .AddCookie(options =>
                {
                    options.Cookie.Name = "ares.console.auth";
                    options.Cookie.HttpOnly = true;
                    options.LoginPath = "/login";
                    options.Events = new CookieAuthenticationEvents
                    {
                        OnValidatePrincipal = async ctx =>
                        {
                            var sid = ctx.Principal?.FindFirst(AresOperatorContext.SessionIdClaim)?.Value;
                            var reg = ctx.HttpContext.RequestServices.GetRequiredService<AresSessionRegistry>();
                            if (!reg.TryGet(sid, out _))
                            {
                                ctx.RejectPrincipal();
                                await ctx.HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
                            }
                        }
                    };
                });
            builder.Services.AddAuthorization();

            var app = builder.Build();
            app.UseAuthentication();
            app.UseAuthorization();
            app.UseAntiforgery();

            app.MapGet("/csrf", (HttpContext ctx, IAntiforgery af) =>
            {
                var tokens = af.GetAndStoreTokens(ctx);
                return Results.Ok(new { field = tokens.FormFieldName, token = tokens.RequestToken });
            }).AllowAnonymous();

            app.MapPost("/auth/login", async (HttpContext ctx, AresAuthService authSvc, IAntiforgery af) =>
            {
                await af.ValidateRequestAsync(ctx);
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

            app.MapPost("/auth/logout", async (HttpContext ctx, AresSessionRegistry reg, IAntiforgery af) =>
            {
                await af.ValidateRequestAsync(ctx);
                var sid = ctx.User.FindFirst(AresOperatorContext.SessionIdClaim)?.Value;
                reg.Remove(sid);
                await ctx.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
                return Results.Redirect("/login");
            }).AllowAnonymous();

            app.MapGet("/protected", (HttpContext ctx) =>
                Results.Ok(new { org = ctx.User.FindFirst(AresOperatorContext.OrgIdClaim)?.Value })
            ).RequireAuthorization();

            await app.StartAsync();
            var addr = app.Services.GetRequiredService<Microsoft.AspNetCore.Hosting.Server.IServer>()
                .Features.Get<IServerAddressesFeature>()!.Addresses.Single();
            return (app, addr);
        }

        var (node1, url1) = await StartRedisBackedNodeAsync();
        var (node2, url2) = await StartRedisBackedNodeAsync();
        try
        {
            var c1 = new CookieContainer();
            using var h1 = new HttpClientHandler { CookieContainer = c1, AllowAutoRedirect = false };
            using var http1 = new HttpClient(h1) { BaseAddress = new Uri(url1) };

            var csrf1 = System.Text.Json.JsonDocument.Parse(await http1.GetStringAsync("/csrf"));
            var loginResp = await http1.PostAsync("/auth/login", new FormUrlEncodedContent(new Dictionary<string, string>
            {
                [csrf1.RootElement.GetProperty("field").GetString()!] = csrf1.RootElement.GetProperty("token").GetString()!,
                ["email"] = "redis@acme.test",
                ["role"] = "Tenant",
                ["org_id"] = "org_acme_fintech",
                ["api_key"] = "ares_tnt_org_acme_fintech_key01",
            }));
            Assert.Equal(HttpStatusCode.Redirect, loginResp.StatusCode);
            Assert.Equal("/", loginResp.Headers.Location?.ToString());

            var c2 = new CookieContainer();
            foreach (Cookie cookie in c1.GetCookies(new Uri(url1)))
            {
                c2.Add(new Uri(url2), new Cookie(cookie.Name, cookie.Value));
            }
            using var h2 = new HttpClientHandler { CookieContainer = c2, AllowAutoRedirect = false };
            using var http2 = new HttpClient(h2) { BaseAddress = new Uri(url2) };

            var node2Ok = await http2.GetAsync("/protected");
            Assert.Equal(HttpStatusCode.OK, node2Ok.StatusCode);
            Assert.Contains("org_acme_fintech", await node2Ok.Content.ReadAsStringAsync());

            var csrf2 = System.Text.Json.JsonDocument.Parse(await http2.GetStringAsync("/csrf"));
            var logoutResp = await http2.PostAsync("/auth/logout", new FormUrlEncodedContent(new Dictionary<string, string>
            {
                [csrf2.RootElement.GetProperty("field").GetString()!] = csrf2.RootElement.GetProperty("token").GetString()!,
            }));
            Assert.Equal(HttpStatusCode.Redirect, logoutResp.StatusCode);

            var node1Revoked = await http1.GetAsync("/protected");
            Assert.Equal(HttpStatusCode.Redirect, node1Revoked.StatusCode);
        }
        finally
        {
            await node1.StopAsync();
            await node2.StopAsync();
        }
    }
}
