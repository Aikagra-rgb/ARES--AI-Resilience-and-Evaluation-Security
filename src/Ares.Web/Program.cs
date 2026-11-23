using System.IO;
using Ares.Web;
using Ares.Web.Services;
using Microsoft.AspNetCore.Antiforgery;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.Extensions.Caching.Distributed;

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Logging.AddConsole();

// Data Protection - use persistent keys in production (or when session persistence is enabled), ephemeral only in development
var dataProtectionKeysPath = builder.Configuration["DataProtection:KeysPath"]
    ?? Path.Combine(builder.Environment.ContentRootPath, "keys");
var persistSessions = builder.Configuration.GetValue<bool?>("Ares:PersistSessions")
    ?? !builder.Environment.IsDevelopment();

if (builder.Environment.IsDevelopment() && !persistSessions)
{
    builder.Services.AddSingleton<IDataProtectionProvider, EphemeralDataProtectionProvider>();
}
else
{
    Directory.CreateDirectory(dataProtectionKeysPath);
    builder.Services.AddDataProtection()
        .PersistKeysToFileSystem(new DirectoryInfo(dataProtectionKeysPath))
        .SetApplicationName("ARES-Security-System");
}

// Configure CORS for Blazor frontend
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins(
                "http://localhost:5000",
                "https://localhost:5001",
                "http://localhost:5149",
                "https://localhost:7149",
                "http://localhost:5180",
                "https://localhost:7180"
            )
            .AllowAnyHeader()
            .AllowAnyMethod()
            .AllowCredentials();
    });
});

var requireHttpsCookie = builder.Configuration.GetValue<bool?>("Ares:RequireHttpsCookie")
    ?? !builder.Environment.IsDevelopment();

// Cookie Authentication & Role/Tenant-Scoped RBAC
builder.Services.AddHttpContextAccessor();
builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
    .AddCookie(options =>
    {
        options.Cookie.Name = "ares.console.auth";
        options.Cookie.HttpOnly = true;
        options.Cookie.SameSite = SameSiteMode.Lax;
        options.Cookie.SecurePolicy = requireHttpsCookie
            ? CookieSecurePolicy.Always
            : CookieSecurePolicy.SameAsRequest;
        options.LoginPath = "/login";
        options.AccessDeniedPath = "/login";
        options.ExpireTimeSpan = TimeSpan.FromHours(8);
        options.SlidingExpiration = true;
        options.Events = new CookieAuthenticationEvents
        {
            OnValidatePrincipal = async context =>
            {
                var sid = context.Principal?.FindFirst(AresOperatorContext.SessionIdClaim)?.Value;
                var registry = context.HttpContext.RequestServices.GetRequiredService<AresSessionRegistry>();
                if (!registry.TryGet(sid, out _))
                {
                    context.RejectPrincipal();
                    await context.HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
                }
            }
        };
    });
builder.Services.AddAuthorization(options =>
{
    options.AddPolicy("AdminOnly", policy => policy.RequireRole("Admin"));
    options.AddPolicy("TenantOrAdmin", policy => policy.RequireAuthenticatedUser());
});
var sessionPersistencePath = persistSessions
    ? Path.Combine(dataProtectionKeysPath, "operator-sessions.protected.json")
    : null;

builder.Services.AddCascadingAuthenticationState();
builder.Services.AddSingleton<AresSessionRegistry>(sp => new AresSessionRegistry(
    dataProtectionProvider: sp.GetRequiredService<IDataProtectionProvider>(),
    persistencePath: sessionPersistencePath,
    distributedCache: sp.GetService<IDistributedCache>(),
    sessionLifetime: TimeSpan.FromHours(8)
));
builder.Services.AddSingleton<IAresSessionStore>(sp => sp.GetRequiredService<AresSessionRegistry>());
builder.Services.AddSingleton<AresAuthService>();
builder.Services.AddScoped<AresOperatorContext>();

var useMock = builder.Configuration.GetValue<bool>("Ares:UseMock", defaultValue: false);
var baseUrl = builder.Configuration["Ares:ApiBaseUrl"] ?? "http://localhost:8000";

// Register named HttpClient for backend communication WITHOUT a global ambient admin API key.
// Every authenticated request must inject its operator-scoped credentials via AresOperatorContext.
builder.Services.AddHttpClient("AresBackend", client =>
{
    client.BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/");
    client.Timeout = TimeSpan.FromSeconds(120);
});

// Add services for Interactive Server only
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents(options =>
    {
        options.DetailedErrors = builder.Environment.IsDevelopment();
    })
    .AddHubOptions(options =>
    {
        // Limit SignalR payload message size to 128KB to prevent buffer exhaustion DoS
        options.MaximumReceiveMessageSize = 128 * 1024;
    });

builder.Services.AddSingleton<MockQdrantCorpusService>();
builder.Services.AddScoped<MockAresApiClient>();

if (useMock)
{
    builder.Services.AddScoped<IAresApiClient>(provider => provider.GetRequiredService<MockAresApiClient>());
    builder.Services.AddSingleton<IQdrantCorpusService>(provider => provider.GetRequiredService<MockQdrantCorpusService>());
}
else
{
    builder.Services.AddScoped<FastApiAresApiClient>(provider =>
    {
        var factory = provider.GetRequiredService<IHttpClientFactory>();
        var http = factory.CreateClient("AresBackend");
        var fallback = provider.GetRequiredService<MockAresApiClient>();
        var opCtx = provider.GetRequiredService<AresOperatorContext>();
        return new FastApiAresApiClient(http, fallback, operatorContext: opCtx);
    });
    builder.Services.AddScoped<IAresApiClient>(provider => provider.GetRequiredService<FastApiAresApiClient>());

    builder.Services.AddScoped<RealQdrantCorpusService>(provider =>
    {
        var factory = provider.GetRequiredService<IHttpClientFactory>();
        var http = factory.CreateClient("AresBackend");
        http.Timeout = TimeSpan.FromSeconds(30);
        var logger = provider.GetService<ILogger<RealQdrantCorpusService>>();
        var opCtx = provider.GetRequiredService<AresOperatorContext>();
        return new RealQdrantCorpusService(http, logger, opCtx);
    });
    builder.Services.AddScoped<IQdrantCorpusService>(provider => provider.GetRequiredService<RealQdrantCorpusService>());
}

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

// Security Headers Middleware (Clickjacking, MIME-sniffing, XSS defense-in-depth, CSP)
app.Use(async (context, next) =>
{
    context.Response.Headers.Append("X-Frame-Options", "DENY");
    context.Response.Headers.Append("X-Content-Type-Options", "nosniff");
    context.Response.Headers.Append("Referrer-Policy", "strict-origin-when-cross-origin");
    context.Response.Headers.Append("Permissions-Policy", "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()");
    context.Response.Headers.Append("Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss: http://localhost:8000 https://*.qdrant.io; img-src 'self' data: https:; font-src 'self'; frame-ancestors 'none';");
    await next();
});

if (!app.Configuration.GetValue<bool>("Ares:DisableHttpsRedirection", defaultValue: false))
{
    app.UseHttpsRedirection();
}
app.UseStaticFiles();
app.UseRouting();
app.UseCors();
app.UseAuthentication();
app.UseAuthorization();
app.UseAntiforgery();

// Cookie sign-in endpoint (CSRF-protected form POST from /login)
app.MapPost("/auth/login", async (HttpContext httpContext, AresAuthService authService, IAntiforgery antiforgery) =>
{
    try
    {
        await antiforgery.ValidateRequestAsync(httpContext);
    }
    catch (AntiforgeryValidationException)
    {
        return Results.BadRequest(new
        {
            error = "invalid_antiforgery_token",
            message = "Missing or invalid CSRF antiforgery token.",
        });
    }

    var form = await httpContext.Request.ReadFormAsync(httpContext.RequestAborted);
    var loginReq = new AresLoginRequest(
        Email: form["email"].ToString(),
        DisplayName: form["display_name"].ToString(),
        Role: form["role"].ToString(),
        OrganizationId: form["org_id"].ToString(),
        ApiKey: form["api_key"].ToString()
    );
    var returnUrl = form["return_url"].ToString();
    if (!AresAuthService.IsSafeLocalReturnUrl(returnUrl))
    {
        returnUrl = "/";
    }

    var result = await authService.AuthenticateAsync(loginReq, httpContext.RequestAborted);
    if (!result.Succeeded || result.Principal is null)
    {
        var err = Uri.EscapeDataString(result.ErrorMessage ?? "Authentication failed.");
        return Results.Redirect($"/login?error={err}&returnUrl={Uri.EscapeDataString(returnUrl)}");
    }

    await httpContext.SignInAsync(
        CookieAuthenticationDefaults.AuthenticationScheme,
        result.Principal,
        new AuthenticationProperties
        {
            IsPersistent = true,
            ExpiresUtc = DateTimeOffset.UtcNow.AddHours(8),
            AllowRefresh = true,
        });

    return Results.Redirect(returnUrl);
}).AllowAnonymous();

// Cookie sign-out endpoint (revokes server-side session & clears HttpOnly cookie)
async Task<IResult> HandleLogoutAsync(HttpContext httpContext, AresSessionRegistry registry)
{
    var sid = httpContext.User.FindFirst(AresOperatorContext.SessionIdClaim)?.Value;
    registry.Remove(sid);
    await httpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
    return Results.Redirect("/login");
}

// GET /auth/logout only cleans up already-evicted/stale cookies (prevents GET CSRF logout of active sessions)
app.MapGet("/auth/logout", async (HttpContext httpContext, AresSessionRegistry registry) =>
{
    var sid = httpContext.User.FindFirst(AresOperatorContext.SessionIdClaim)?.Value;
    if (httpContext.User.Identity?.IsAuthenticated == true && registry.TryGet(sid, out _))
    {
        return Results.Redirect("/");
    }

    await httpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
    return Results.Redirect("/login");
}).AllowAnonymous();

app.MapPost("/auth/logout", async (HttpContext httpContext, AresSessionRegistry registry, IAntiforgery antiforgery) =>
{
    try
    {
        await antiforgery.ValidateRequestAsync(httpContext);
    }
    catch (AntiforgeryValidationException)
    {
        return Results.BadRequest(new
        {
            error = "invalid_antiforgery_token",
            message = "Missing or invalid CSRF antiforgery token.",
        });
    }

    return await HandleLogoutAsync(httpContext, registry);
}).AllowAnonymous();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();

public partial class Program { }
