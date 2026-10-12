using Ares.Web;
using Ares.Web.Services;
using Microsoft.AspNetCore.DataProtection;
using System.IO;

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Logging.AddConsole();

// Data Protection - use persistent keys in production, ephemeral only in development
var dataProtectionKeysPath = builder.Configuration["DataProtection:KeysPath"] 
    ?? Path.Combine(builder.Environment.ContentRootPath, "keys");

if (builder.Environment.IsDevelopment())
{
    // Development: ephemeral is fine
    builder.Services.AddSingleton<IDataProtectionProvider, EphemeralDataProtectionProvider>();
}
else
{
    // Production: persist keys to filesystem
    Directory.CreateDirectory(dataProtectionKeysPath);
    builder.Services.AddDataProtection()
        .PersistKeysToFileSystem(new DirectoryInfo(dataProtectionKeysPath))
        .SetApplicationName("ARES-Security-System");
}

// Configure CORS for Blazor WebAssembly frontend
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(builder =>
    {
        builder.WithOrigins(
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

// Register Ares API clients
builder.Services.AddHttpClient<FastApiAresApiClient>(client =>
{
    var baseUrl = builder.Configuration["Ares:ApiBaseUrl"] ?? "http://localhost:8000";
    client.BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/");
    client.Timeout = TimeSpan.FromSeconds(120);
});
builder.Services.AddScoped<IAresApiClient, FastApiAresApiClient>();

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

var useMock = builder.Configuration.GetValue<bool>("Ares:UseMock", defaultValue: false);
var baseUrl = builder.Configuration["Ares:ApiBaseUrl"] ?? "http://localhost:8000";
var apiKey = builder.Configuration["Ares:ApiKey"] ?? "ares-dev-secret-key-42";

if (useMock)
{
    builder.Services.AddScoped<IAresApiClient>(provider => provider.GetRequiredService<MockAresApiClient>());
    builder.Services.AddSingleton<IQdrantCorpusService>(provider => provider.GetRequiredService<MockQdrantCorpusService>());
}
else
{
    builder.Services.AddHttpClient<RealQdrantCorpusService>(client =>
    {
        client.BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/");
        client.Timeout = TimeSpan.FromSeconds(30);
        client.DefaultRequestHeaders.Add("X-Ares-Api-Key", apiKey);
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

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.UseCors();
app.UseAntiforgery();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
