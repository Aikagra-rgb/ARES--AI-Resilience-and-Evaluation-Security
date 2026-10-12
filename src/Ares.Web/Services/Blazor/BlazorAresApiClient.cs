using System.Collections.Concurrent;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Ares.Web.Models;

namespace Ares.Web.Services;

/// <summary>
/// HTTP implementation for Blazor WebAssembly frontend, with HMAC request signing.
/// </summary>
public sealed class BlazorAresApiClient : IAresApiClient
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };
    static BlazorAresApiClient() => JsonOptions.Converters.Add(new JsonStringEnumConverter());

    private readonly HttpClient _http;
    private readonly MockAresApiClient _learningFallback;
    private readonly ConcurrentDictionary<string, ArenaTestConfiguration> _submittedConfigurations = new();
    private readonly ILogger<BlazorAresApiClient> _logger;

    // HMAC signing support
    private readonly string _signingKey;
    private readonly string _keyId;

    /// <summary>
    /// Default constructor for dependency injection.
    /// </summary>
    public BlazorAresApiClient(HttpClient http, MockAresApiClient learningFallback, ILogger<BlazorAresApiClient> logger, string signingKey = null, string keyId = "ares-web-blazor")
    {
        _http = http;
        _learningFallback = learningFallback;
        _logger = logger;
        _signingKey = signingKey;
        _keyId = keyId;
    }

    /// <summary>
    /// Compute HMAC-SHA256 signature for the request.
    /// </summary>
    /// <param name="method">HTTP method</param>
    /// <param name="path">Request path</param>
    /// <param name="queryParams">Query parameters</param>
    /// <param name="body">Request body</param>
    /// <returns>(timestamp, nonce, signature, keyId)</returns>
    private (string timestamp, string nonce, string signature, string keyId) ComputeSignature(string method, string path, Dictionary<string, string> queryParams, string body)
    {
        if (string.IsNullOrEmpty(_signingKey))
        {
            // No signing key configured - operate without signature
            var ts = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString();
            var nc = Guid.NewGuid().ToString("N");
            return (ts, nc, "", _keyId);
        }

        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString();
        var nonce = Guid.NewGuid().ToString("N");
        var sortedQuery = string.IsNullOrEmpty(queryParams) || queryParams.Count == 0 ? "" : $"?{string.Join("&", queryParams.OrderBy(k => k.Key).Select(k => $"{k.Key}={k.Value}"))}";

        // Create canonical request
        var bodyHash = string.IsNullOrEmpty(body) ? "" : BitConverter.ToString(SHA256.HashData(Encoding.UTF8.GetBytes(body))).Replace("-", "").ToLowerInvariant();

        var canonical = $"{method.ToUpperInvariant()}\n{path}\n{sortedQuery}\n{bodyHash}\n{timestamp}\n{nonce}";

        var signature = BitConverter.ToString(SHA256.HashData(Encoding.UTF8.GetBytes(canonical))).Replace("-", "").ToLowerInvariant();

        return (timestamp, nonce, signature, _keyId);
    }

    /// <summary>
    /// Create a signed HTTP request message.
    /// </summary>
    /// <param name="method">HTTP method</param>
    /// <param name="url">Full URL</param>
    /// <param name="body">Request body (null for GET)</param>
    /// <returns>Signed HttpRequestMessage</returns>
    private HttpRequestMessage CreateSignedRequest(string method, string url, string body = null)
    {
        var uri = new Uri(url);
        var request = new HttpRequestMessage(new HttpMethod(method), uri);

        // Prepare query parameters dictionary
        var queryParams = new Dictionary<string, string>();

        // If the URL has query parameters, parse them
        if (uri.Query.Length > 1)
        {
            var queryStrings = uri.Query.TrimStart('?').Split('&');
            foreach (var qs in queryStrings)
            {
                var parts = qs.Split('=');
                if (parts.Length >= 2)
                {
                    queryParams[Uri.UnescapeDataString(parts[0])] = Uri.UnescapeDataString(parts[1]);
                }
            }
            // Rebuild URL without query string for signing
            request.RequestUri = new Uri($"{uri.Scheme}://{uri.Authority}{uri.PathAndQuery}");
        }
        else
        {
            request.RequestUri = uri;
        }

        // Set body if provided
        if (!string.IsNullOrEmpty(body))
        {
            request.Content = new StringContent(body, Encoding.UTF8, "application/json");
        }

        // Compute signature
        var (timestamp, nonce, signature, keyId) = ComputeSignature(method, uri.PathAndQuery, queryParams, body);

        // Add signature headers
        request.Headers.Add("X-Ares-Timestamp", timestamp);
        request.Headers.Add("X-Ares-Nonce", nonce);
        if (!string.IsNullOrEmpty(signature))
        {
            request.Headers.Add("X-Ares-Signature", signature);
            request.Headers.Add("X-Ares-Key-Id", keyId);
        }

        _logger.LogDebug("Created signed request: {Method} {Url} with timestamp={Ts}, nonce={Nonce}", method, url, timestamp, nonce);

        return request;
    }

    /// <summary>
    /// Send a signed request and read the response.
    /// </summary>
    /// <typeparam name="T">Response type</typeparam>
    /// <param name="method">HTTP method</param>
    /// <param name="url">Request URL</param>
    /// <param name="body">Request body (null for GET)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Response</returns>
    private async Task<T> SendSignedAsync<T>(string method, string url, string body = null, CancellationToken cancellationToken = default)
    {
        var request = CreateSignedRequest(method, url, body);
        var response = await _http.SendAsync(request, cancellationToken);

        if (response.IsSuccessStatusCode)
        {
            var responseBody = await response.Content.ReadAsStringAsync(cancellationToken);
            return JsonSerializer.Deserialize<T>(responseBody, JsonOptions);
        }

        var detail = await response.Content.ReadAsStringAsync(cancellationToken);
        throw new HttpRequestException($"ARES API returned {(int)response.StatusCode}: {detail}", null, response.StatusCode);
    }

    // ── API Methods ──

    public async Task<CreateTestResponse> CreateTestAsync(CreateTestRequest request, CancellationToken cancellationToken)
    {
        var body = JsonSerializer.Serialize(new { configuration = request.Configuration }, JsonOptions);
        return await SendSignedAsync<ApiCreateTestResponse>("POST", "api/v1/tests", body, cancellationToken)
            .ContinueWith(t => new CreateTestResponse(t.Result.TestId, ParseStatus(t.Result.Status), t.Result.CorrelationId, t.Result.CreatedAt));
    }

    public async Task<TestRun?> GetTestAsync(string testId, CancellationToken cancellationToken)
    {
        return await SendSignedAsync<ApiTestRun>("GET", $"api/v1/tests/{testId}", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result != null ? ToTestRun(t.Result) : null);
    }

    public async Task<CancelTestResponse> CancelTestAsync(string testId, CancellationToken cancellationToken)
    {
        return await SendSignedAsync<ApiCancelTestResponse>("POST", $"api/v1/tests/{testId}/cancel", cancellationToken: cancellationToken)
            .ContinueWith(t => new CancelTestResponse(t.Result.TestId, ParseStatus(t.Result.Status), t.Result.CorrelationId));
    }

    public async Task<IReadOnlyList<RecentTestItem>> GetRecentTestsAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<ApiRecentTests>("GET", "api/v1/tests/recent", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Items.Select(item => new RecentTestItem(item.Id, item.Timestamp, ParseCategory(item.Category), ParseProvider(item.Provider), item.Model, item.RiskScore, ParseStatus(item.Status))).ToList());
    }

    public async Task<DashboardSummary> GetDashboardSummaryAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<ApiDashboardSummary>("GET", "api/v1/dashboard/summary", cancellationToken: cancellationToken)
            .ContinueWith(t => new DashboardSummary(t.Result.Metrics.Select(metric => new MetricValue(metric.Label, metric.Value, metric.Change, metric.Trend, ParseSeverity(metric.Status), metric.Description)).ToList(), t.Result.CorpusSize, t.Result.ProtectedApplications, t.Result.GeneratedAt, t.Result.IsPartial));
    }

    public async Task<IReadOnlyList<TrendPoint>> GetDashboardTrendsAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<IReadOnlyList<ApiTrends>>("GET", "api/v1/dashboard/trends", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Points.Select(point => new TrendPoint(point.Date, point.Tested, point.Blocked, point.Successful, point.Incidents)).ToList());
    }

    public async Task<IReadOnlyList<CategoryMetric>> GetCategoryMetricsAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<IReadOnlyList<ApiCategories>>("GET", "api/v1/dashboard/categories", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Items.Select(item => new CategoryMetric(ParseCategory(item.Category), item.Tests, item.Successful, item.Blocked, item.SuccessRate)).ToList());
    }

    public async Task<IReadOnlyList<RuntimeIncident>> GetRecentIncidentsAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<IReadOnlyList<ApiIncidents>>("GET", "api/v1/incidents/recent", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Items.Select(item => new RuntimeIncident(item.Id, item.Application, ParseCategory(item.Category), ParseSeverity(item.Severity), item.DetectedAt, item.EnforcementAction, Enum.Parse<IncidentStatus>(item.Status, true), item.CorrelationId)).ToList());
    }

    public async Task<HardeningComparison> GetHardeningComparisonAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<ApiHardening>("GET", "api/v1/hardening/comparison", cancellationToken: cancellationToken)
            .ContinueWith(t => new HardeningComparison(t.Result.BaselineSuccessRate, t.Result.HardenedSuccessRate, t.Result.ImprovementPoints, t.Result.TestsIncluded, t.Result.LastCycle ?? DateOnly.MinValue));
    }

    public async Task<IReadOnlyList<ProviderHealth>> GetProviderHealthAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<IReadOnlyList<ApiProviders>>("GET", "api/v1/providers/health", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Items.Select(item => new ProviderHealth(string.IsNullOrEmpty(item.Provider) ? null : ParseProvider(item.Provider), item.Name, Enum.Parse<ProviderStatus>(item.Status, true), item.Detail, item.CheckedAt)).ToList());
    }

    public async Task<IReadOnlyList<ModelConfiguration>> GetModelsAsync(AiProvider provider, CancellationToken cancellationToken)
    {
        var query = new Dictionary<string, string> { ["provider"] = provider.ToString() };
        return await SendSignedAsync<IReadOnlyList<ApiModels>>("GET", $"api/v1/models?{string.Join("&", query.Select(k => $"{k.Key}={k.Value}"))}", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Items.Select(item => new ModelConfiguration(ParseProvider(item.Provider), item.Id, item.DisplayName, item.ContextWindow, item.IsAvailable)).ToList());
    }

    public async Task<ChallengePageResult> GetChallengesAsync(ChallengeTrack? track, DifficultyTier? tier, AttackCategory? category, string? search, int page, CancellationToken cancellationToken)
    {
        var query = new List<string> { ["page"] = page.ToString() };
        if (track.HasValue) query.Add(["track"] = track.Value);
        if (tier.HasValue) query.Add(["tier"] = tier.Value);
        if (category.HasValue) query.Add(["category"] = category.Value);
        if (!string.IsNullOrWhiteSpace(search)) query.Add(["search"] = search);

        return await SendSignedAsync<ApiChallengePage>($"GET", $"api/v1/arena/challenges?{string.Join("&", query)}", cancellationToken: cancellationToken)
            .ContinueWith(t => new ChallengePageResult(t.Result.Items.Select(ToChallenge).ToList(), t.Result.Progress.Select(ToProgress).ToList(), t.Result.TotalCount, t.Result.Page, t.Result.PageSize));
    }

    public async Task<Challenge?> GetChallengeAsync(string challengeId, CancellationToken cancellationToken)
    {
        return await SendSignedAsync<ApiChallenge>($"GET", $"api/v1/arena/challenges/{challengeId}", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result != null ? ToChallenge(t.Result) : null);
    }

    public async Task<IReadOnlyList<ChallengeRoom>> GetRoomsAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<IReadOnlyList<ApiRooms>>("GET", "api/v1/arena/rooms", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Items.Select(ToRoom).ToList());
    }

    public async Task<ChallengeRoom?> GetRoomAsync(string roomId, CancellationToken cancellationToken)
    {
        return await SendSignedAsync<ApiRoom>($"GET", $"api/v1/arena/rooms/{roomId}", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result != null ? ToRoom(t.Result) : null);
    }

    public async Task<IReadOnlyList<LearningPath>> GetLearningPathsAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<IReadOnlyList<ApiPaths>>("GET", "api/v1/arena/paths", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Items.Select(item => new LearningPath(item.Id, item.Name, item.Description, item.RoomIds, Enum.Parse<BadgeType>(item.BadgeAwarded, true), item.Colour)).ToList());
    }

    public async Task<SubmitChallengeResponse> SubmitChallengeAsync(SubmitChallengeRequest request, CancellationToken cancellationToken)
    {
        var body = JsonSerializer.Serialize(new { challenge_id = request.ChallengeId, test_run_id = request.TestRunId }, JsonOptions);
        return await SendSignedAsync<ApiSubmitChallenge>("POST", "api/v1/arena/submissions", body, cancellationToken)
            .ContinueWith(t => new SubmitChallengeResponse(ToSubmission(t.Result.Submission), t.Result.BadgeUnlocked, null, t.Result.NextChallengeId));
    }

    public async Task<IReadOnlyList<ChallengeSubmission>> GetMySubmissionsAsync(string challengeId, CancellationToken cancellationToken)
    {
        return await SendSignedAsync<IReadOnlyList<ApiSubmissions>>("GET", $"api/v1/arena/challenges/{challengeId}/submissions", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Items.Select(ToSubmission).ToList());
    }

    public async Task<IReadOnlyList<ChallengeProgressItem>> GetMyProgressAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<IReadOnlyList<ApiProgressList>>("GET", "api/v1/arena/progress", cancellationToken: cancellationToken)
            .ContinueWith(t => t.Result.Items.Select(ToProgress).ToList());
    }

    public async Task<UserProfile> GetMyProfileAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<ApiProfile>("GET", "api/v1/arena/profile", cancellationToken: cancellationToken)
            .ContinueWith(t => new UserProfile(t.Result.Id, t.Result.DisplayName, t.Result.Initials, t.Result.XpTotal, t.Result.Level, t.Result.XpThisLevel, t.Result.XpToNextLevel, [], t.Result.ChallengesSolved, t.Result.AttackerSolved, t.Result.DefenderSolved, t.Result.MemberSince));
    }

    // Phase 6: ML Stats, Semantic Search, Prompt Hardening, Runtime Gateway

    public async Task<StatsResponse> GetStatsAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<StatsResponse>("GET", "api/v1/stats", cancellationToken: cancellationToken);
    }

    public async Task<StatsResponse> RetrainStatsAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<StatsResponse>("POST", "api/stats/retrain", cancellationToken: cancellationToken);
    }

    public async Task<string> GetReportAsync(string testId, string format, CancellationToken cancellationToken)
    {
        var route = $"api/tests/{testId}/report?format={format}";
        try
        {
            using var response = await _http.SendAsync(new HttpRequestMessage(HttpMethod.Get, route) { });
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadAsStringAsync(cancellationToken);
            var detail = await response.Content.ReadAsStringAsync(cancellationToken);
            throw new HttpRequestException($"ARES API '{route}' returned {(int)response.StatusCode}: {detail}", null, response.StatusCode);
        }
        catch (HttpRequestException ex) when (ex.StatusCode is null)
        {
            throw new HttpRequestException(ex.Message, ex.InnerException, ex.StatusCode ?? System.Net.HttpStatusCode.InternalServerError);
        }
    }

    public async Task<SearchResponse> SearchAsync(SearchRequest request, CancellationToken cancellationToken)
    {
        var body = JsonSerializer.Serialize(request, JsonOptions);
        return await SendSignedAsync<SearchResponse>("POST", "api/v1/search", body, cancellationToken);
    }

    public async Task<HardenResponse> HardenAsync(HardenRequest request, CancellationToken cancellationToken)
    {
        var body = JsonSerializer.Serialize(request, JsonOptions);
        return await SendSignedAsync<HardenResponse>("POST", "api/v1/harden", body, cancellationToken);
    }

    public async Task<GatewayEnforceResponse> EnforcePromptAsync(GatewayEnforceRequest request, CancellationToken cancellationToken)
    {
        var body = JsonSerializer.Serialize(request, JsonOptions);
        return await SendSignedAsync<GatewayEnforceResponse>("POST", "api/gateway/enforce", body, cancellationToken);
    }

    public async Task<GatewayStatusResponse> GetGatewayStatusAsync(CancellationToken cancellationToken)
    {
        return await SendSignedAsync<GatewayStatusResponse>("GET", "api/gateway/status", cancellationToken: cancellationToken);
    }
}