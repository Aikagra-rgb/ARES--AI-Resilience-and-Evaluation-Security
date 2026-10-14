using System.Collections.Concurrent;
using System.Net;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Ares.Web.Models;

namespace Ares.Web.Services;

/// <summary>HTTP implementation for dashboard, controlled tests, and persisted Arena learning progress.</summary>
public sealed class FastApiAresApiClient : IAresApiClient
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };
    static FastApiAresApiClient() => JsonOptions.Converters.Add(new JsonStringEnumConverter());
    
    private readonly HttpClient _http;
    private readonly MockAresApiClient _learningFallback;
    private readonly ConcurrentDictionary<string, ArenaTestConfiguration> _submittedConfigurations = new();
    private readonly string? _signingKey;
    private readonly string _keyId;

    public FastApiAresApiClient(HttpClient http, MockAresApiClient learningFallback, string? signingKey = null, string keyId = "ares-web")
    {
        _http = http;
        _learningFallback = learningFallback;
        _signingKey = signingKey;
        _keyId = keyId;
    }

    private (string timestamp, string nonce, string signature, string keyId) ComputeSignature(HttpRequestMessage request, string body = "")
    {
        if (string.IsNullOrEmpty(_signingKey))
        {
            return (DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString(), Guid.NewGuid().ToString("N"), "", _keyId);
        }

        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds().ToString();
        var nonce = Guid.NewGuid().ToString("N");
        var method = request.Method.Method;
        var path = request.RequestUri?.AbsolutePath ?? string.Empty;
        var query = request.RequestUri?.Query.TrimStart('?') ?? string.Empty;
        
        // Create canonical request
        var canonicalParts = new List<string>
        {
            method.ToUpperInvariant(),
            path,
            query,
            ComputeSha256(body),
            timestamp,
            nonce
        };
        var canonical = string.Join("\n", canonicalParts);
        
        var signature = ComputeHmacSha256(canonical, _signingKey);
        return (timestamp, nonce, signature, _keyId);
    }

    private string ComputeSha256(string input)
    {
        if (string.IsNullOrEmpty(input))
            return "";
        
        using var sha256 = SHA256.Create();
        var hash = sha256.ComputeHash(Encoding.UTF8.GetBytes(input));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private string ComputeHmacSha256(string message, string key)
    {
        using var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(key));
        var hash = hmac.ComputeHash(Encoding.UTF8.GetBytes(message));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private HttpRequestMessage CreateSignedRequest(HttpMethod method, string url, object? body = null)
    {
        var request = new HttpRequestMessage(method, url);
        
        string bodyJson = "";
        if (body != null)
        {
            bodyJson = JsonSerializer.Serialize(body, JsonOptions);
            request.Content = new StringContent(bodyJson, Encoding.UTF8, "application/json");
        }
        
        var (timestamp, nonce, signature, keyId) = ComputeSignature(request, bodyJson);
        
        request.Headers.Add("X-Ares-Timestamp", timestamp);
        request.Headers.Add("X-Ares-Nonce", nonce);
        if (!string.IsNullOrEmpty(signature))
        {
            request.Headers.Add("X-Ares-Signature", signature);
            request.Headers.Add("X-Ares-Key-Id", keyId);
        }
        
        return request;
    }

    private async Task<HttpResponseMessage> SendSignedAsync(HttpMethod method, string url, object? body = null, CancellationToken cancellationToken = default)
    {
        var request = CreateSignedRequest(method, url, body);
        return await _http.SendAsync(request, cancellationToken);
    }

    private async Task<T> SendSignedAsync<T>(HttpMethod method, string url, object? body = null, CancellationToken cancellationToken = default)
    {
        using var response = await SendSignedAsync(method, url, body, cancellationToken);
        if (response.IsSuccessStatusCode)
        {
            return await response.Content.ReadFromJsonAsync<T>(JsonOptions, cancellationToken)
                ?? throw new InvalidOperationException($"ARES API returned an empty response for '{url}'.");
        }

        var detail = await response.Content.ReadAsStringAsync(cancellationToken);
        throw new HttpRequestException(
            $"ARES API '{url}' returned {(int)response.StatusCode}: {detail}", null, response.StatusCode);
    }

    public async Task<CreateTestResponse> CreateTestAsync(CreateTestRequest request, CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Post, "api/v1/tests", new { configuration = request.Configuration }, cancellationToken);
        var body = await ReadAsync<ApiCreateTestResponse>(response, cancellationToken);
        _submittedConfigurations[body.TestId] = CopyConfiguration(request.Configuration);
        return new CreateTestResponse(body.TestId, ParseStatus(body.Status), body.CorrelationId, body.CreatedAt);
    }

    public async Task<TestRun?> GetTestAsync(string testId, CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, $"api/v1/tests/{Uri.EscapeDataString(testId)}", cancellationToken: cancellationToken);
        if (response.StatusCode == HttpStatusCode.NotFound) return null;
        var body = await ReadAsync<ApiTestRun>(response, cancellationToken);
        return ToTestRun(body);
    }

    public async Task<TestRun> SimulateTestAsync(string testId, IProgress<ExecutionProgress>? progress, CancellationToken cancellationToken)
    {
        // Legacy interface name retained while the UI transitions to server-side execution polling.
        var started = DateTimeOffset.UtcNow;
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var run = await GetTestAsync(testId, cancellationToken) ?? throw new KeyNotFoundException("Test run was not found.");
            var percent = run.Status switch
            {
                TestRunStatus.Queued => 10,
                TestRunStatus.Running => 55,
                TestRunStatus.Completed or TestRunStatus.Blocked => 100,
                _ => 0
            };
            progress?.Report(new ExecutionProgress(percent, run.Status.ToLabel(), StatusMessage(run), run.TokenEstimate, run.DurationMilliseconds));
            if (run.Status is TestRunStatus.Completed or TestRunStatus.Blocked or TestRunStatus.Failed or TestRunStatus.Cancelled) return run;
            if (DateTimeOffset.UtcNow - started > TimeSpan.FromMinutes(3)) throw new TimeoutException("The server-side test did not complete in time.");
            await Task.Delay(TimeSpan.FromSeconds(1), cancellationToken);
        }
    }

    public async Task<CancelTestResponse> CancelTestAsync(string testId, CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Post, $"api/v1/tests/{Uri.EscapeDataString(testId)}/cancel", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiCancelTestResponse>(response, cancellationToken);
        return new CancelTestResponse(body.TestId, ParseStatus(body.Status), body.CorrelationId);
    }

    public async Task<IReadOnlyList<RecentTestItem>> GetRecentTestsAsync(CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, "api/v1/tests/recent", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiRecentTests>(response, cancellationToken);
        return body.Items.Select(item => new RecentTestItem(item.Id, item.Timestamp, ParseCategory(item.Category), ParseProvider(item.Provider), item.Model, item.RiskScore, ParseStatus(item.Status))).ToList();
    }

    public Task<CorpusSaveResponse> SaveAttackToCorpusAsync(CorpusSaveRequest request, CancellationToken cancellationToken) =>
        throw new NotSupportedException("Attack-corpus persistence is not implemented by the API yet.");

    public async Task<DashboardSummary> GetDashboardSummaryAsync(CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, "api/v1/dashboard/summary", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiDashboardSummary>(response, cancellationToken);
        return new DashboardSummary(body.Metrics.Select(metric => new MetricValue(metric.Label, metric.Value, metric.Change, metric.Trend, ParseSeverity(metric.Status), metric.Description)).ToList(), body.CorpusSize, body.ProtectedApplications, body.GeneratedAt, body.IsPartial);
    }

    public async Task<IReadOnlyList<TrendPoint>> GetDashboardTrendsAsync(CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, "api/v1/dashboard/trends", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiTrends>(response, cancellationToken);
        return body.Points.Select(point => new TrendPoint(point.Date, point.Tested, point.Blocked, point.Successful, point.Incidents)).ToList();
    }

    public async Task<IReadOnlyList<CategoryMetric>> GetCategoryMetricsAsync(CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, "api/v1/dashboard/categories", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiCategories>(response, cancellationToken);
        return body.Items.Select(item => new CategoryMetric(ParseCategory(item.Category), item.Tests, item.Successful, item.Blocked, item.SuccessRate)).ToList();
    }

    public async Task<IReadOnlyList<RuntimeIncident>> GetRecentIncidentsAsync(CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, "api/v1/incidents/recent", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiIncidents>(response, cancellationToken);
        return body.Items.Select(item => new RuntimeIncident(item.Id, item.Application, ParseCategory(item.Category), ParseSeverity(item.Severity), item.DetectedAt, item.EnforcementAction, Enum.Parse<IncidentStatus>(item.Status, true), item.CorrelationId)).ToList();
    }

    public async Task<HardeningComparison> GetHardeningComparisonAsync(CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, "api/v1/hardening/comparison", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiHardening>(response, cancellationToken);
        return new HardeningComparison(body.BaselineSuccessRate, body.HardenedSuccessRate, body.ImprovementPoints, body.TestsIncluded, body.LastCycle ?? DateOnly.MinValue);
    }

    public async Task<IReadOnlyList<ProviderHealth>> GetProviderHealthAsync(CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, "api/v1/providers/health", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiProviders>(response, cancellationToken);
        return body.Items.Select(item => new ProviderHealth(string.IsNullOrEmpty(item.Provider) ? null : ParseProvider(item.Provider), item.Name, Enum.Parse<ProviderStatus>(item.Status, true), item.Detail, item.CheckedAt)).ToList();
    }

    public async Task<IReadOnlyList<ModelConfiguration>> GetModelsAsync(AiProvider provider, CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, $"api/v1/models?provider={Uri.EscapeDataString(provider.ToString())}", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiModels>(response, cancellationToken);
        return body.Items.Select(item => new ModelConfiguration(ParseProvider(item.Provider), item.Id, item.DisplayName, item.ContextWindow, item.IsAvailable)).ToList();
    }

    public async Task<ChallengePageResult> GetChallengesAsync(ChallengeTrack? track, DifficultyTier? tier, AttackCategory? category, string? search, int page, CancellationToken cancellationToken)
    {
        var query = new List<string> { $"page={page}" };
        if (track.HasValue) query.Add($"track={track.Value}");
        if (tier.HasValue) query.Add($"tier={tier.Value}");
        if (category.HasValue) query.Add($"category={category.Value}");
        if (!string.IsNullOrWhiteSpace(search)) query.Add($"search={Uri.EscapeDataString(search)}");
        using var response = await SendSignedAsync(HttpMethod.Get, $"api/v1/arena/challenges?{string.Join('&', query)}", cancellationToken: cancellationToken);
        var body = await ReadAsync<ApiChallengePage>(response, cancellationToken);
        return new ChallengePageResult(body.Items.Select(ToChallenge).ToList(), body.Progress.Select(ToProgress).ToList(), body.TotalCount, body.Page, body.PageSize);
    }

    public async Task<Challenge?> GetChallengeAsync(string challengeId, CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, $"api/v1/arena/challenges/{Uri.EscapeDataString(challengeId)}", cancellationToken: cancellationToken);
        if (response.StatusCode == HttpStatusCode.NotFound) return null;
        return ToChallenge(await ReadAsync<ApiChallenge>(response, cancellationToken));
    }

    public async Task<IReadOnlyList<ChallengeRoom>> GetRoomsAsync(CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, "api/v1/arena/rooms", cancellationToken: cancellationToken);
        return (await ReadAsync<ApiRooms>(response, cancellationToken)).Items.Select(ToRoom).ToList();
    }

    public async Task<ChallengeRoom?> GetRoomAsync(string roomId, CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, $"api/v1/arena/rooms/{Uri.EscapeDataString(roomId)}", cancellationToken: cancellationToken);
        if (response.StatusCode == HttpStatusCode.NotFound) return null;
        return ToRoom(await ReadAsync<ApiRoom>(response, cancellationToken));
    }

    public async Task<IReadOnlyList<LearningPath>> GetLearningPathsAsync(CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Get, "api/v1/arena/paths", cancellationToken: cancellationToken);
        return (await ReadAsync<ApiPaths>(response, cancellationToken)).Items.Select(item => new LearningPath(item.Id, item.Name, item.Description, item.RoomIds, Enum.Parse<BadgeType>(item.BadgeAwarded, true), item.Colour)).ToList();
    }

    public async Task<SubmitChallengeResponse> SubmitChallengeAsync(SubmitChallengeRequest request, CancellationToken cancellationToken)
    {
        using var response = await SendSignedAsync(HttpMethod.Post, "api/v1/arena/submissions", new { challenge_id = request.ChallengeId, test_run_id = request.TestRunId }, cancellationToken);
        var body = await ReadAsync<ApiSubmitChallenge>(response, cancellationToken);
        return new SubmitChallengeResponse(ToSubmission(body.Submission), body.BadgeUnlocked, null, body.NextChallengeId);
    }

    public async Task<IReadOnlyList<ChallengeSubmission>> GetMySubmissionsAsync(string challengeId, CancellationToken cancellationToken)
    {
        using var response = await _http.GetAsync($"api/v1/arena/challenges/{Uri.EscapeDataString(challengeId)}/submissions", cancellationToken);
        return (await ReadAsync<ApiSubmissions>(response, cancellationToken)).Items.Select(ToSubmission).ToList();
    }

    public async Task<IReadOnlyList<ChallengeProgressItem>> GetMyProgressAsync(CancellationToken cancellationToken)
    {
        using var response = await _http.GetAsync("api/v1/arena/progress", cancellationToken);
        return (await ReadAsync<ApiProgressList>(response, cancellationToken)).Items.Select(ToProgress).ToList();
    }

    public async Task<UserProfile> GetMyProfileAsync(CancellationToken cancellationToken)
    {
        using var response = await _http.GetAsync("api/v1/arena/profile", cancellationToken);
        var item = await ReadAsync<ApiProfile>(response, cancellationToken);
        return new UserProfile(item.Id, item.DisplayName, item.Initials, item.XpTotal, item.Level, item.XpThisLevel, item.XpToNextLevel, [], item.ChallengesSolved, item.AttackerSolved, item.DefenderSolved, item.MemberSince);
    }
    public Task<IReadOnlyList<OrgAssessment>> GetOrgAssessmentsAsync(CancellationToken cancellationToken) => _learningFallback.GetOrgAssessmentsAsync(cancellationToken);
    public Task<OrgAssessment> CreateOrgAssessmentAsync(CreateOrgAssessmentRequest request, CancellationToken cancellationToken) => _learningFallback.CreateOrgAssessmentAsync(request, cancellationToken);
    public Task<Challenge> CreateChallengeAsync(CreateChallengeRequest request, CancellationToken cancellationToken) => _learningFallback.CreateChallengeAsync(request, cancellationToken);
    public Task<Challenge> UpdateChallengeAsync(UpdateChallengeRequest request, CancellationToken cancellationToken) => _learningFallback.UpdateChallengeAsync(request, cancellationToken);

    // ── Phase 6: ML Stats, Semantic Search, Prompt Hardening, Runtime Gateway ──
    // These endpoints fail loudly. If the backend is down or returns an error, the caller
    // receives an exception that the page shows to the user, instead of silently being
    // handed mock data that looks like a real result.
    public Task<StatsResponse> GetStatsAsync(CancellationToken cancellationToken) =>
        SendLiveAsync<StatsResponse>(HttpMethod.Get, ["api/v1/stats", "api/stats"], null, cancellationToken);

    public Task<StatsResponse> RetrainStatsAsync(CancellationToken cancellationToken) =>
        SendLiveAsync<StatsResponse>(HttpMethod.Post, ["api/stats/retrain", "api/v1/stats/retrain"], null, cancellationToken);

    public async Task<string> GetReportAsync(string testId, string format, CancellationToken cancellationToken)
    {
        var route = $"api/tests/{Uri.EscapeDataString(testId)}/report?format={Uri.EscapeDataString(format)}";
        try
        {
            using var response = await _http.GetAsync(route, cancellationToken);
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadAsStringAsync(cancellationToken);
            var detail = await response.Content.ReadAsStringAsync(cancellationToken);
            throw new HttpRequestException($"ARES API '{route}' returned {(int)response.StatusCode}: {Truncate(detail)}", null, response.StatusCode);
        }
        catch (HttpRequestException ex) when (ex.StatusCode is null)
        {
            throw UnreachableError(ex);
        }
    }

    public Task<SearchResponse> SearchAsync(SearchRequest request, CancellationToken cancellationToken) =>
        SendLiveAsync<SearchResponse>(HttpMethod.Post, ["api/v1/search", "api/search"], request, cancellationToken);

    public Task<HardenResponse> HardenAsync(HardenRequest request, CancellationToken cancellationToken) =>
        SendLiveAsync<HardenResponse>(HttpMethod.Post, ["api/v1/harden", "api/harden"], request, cancellationToken);

    public Task<GatewayEnforceResponse> EnforcePromptAsync(GatewayEnforceRequest request, CancellationToken cancellationToken) =>
        SendLiveAsync<GatewayEnforceResponse>(HttpMethod.Post, ["api/gateway/enforce"], request, cancellationToken);

    public Task<GatewayStatusResponse> GetGatewayStatusAsync(CancellationToken cancellationToken) =>
        SendLiveAsync<GatewayStatusResponse>(HttpMethod.Get, ["api/gateway/status"], null, cancellationToken);

    public Task<OptimizePromptResponse> OptimizePromptAsync(OptimizePromptRequest request, CancellationToken cancellationToken) =>
        SendLiveAsync<OptimizePromptResponse>(HttpMethod.Post, ["api/v1/optimizer/optimize", "api/optimizer/optimize"], request, cancellationToken);

    public Task<IReadOnlyList<IntentInfoResponse>> GetIntentCategoriesAsync(CancellationToken cancellationToken) =>
        SendLiveAsync<IReadOnlyList<IntentInfoResponse>>(HttpMethod.Get, ["api/v1/optimizer/intents", "api/optimizer/intents"], null, cancellationToken);


    /// <summary>
    /// Calls the first route that exists (alternates are only tried on HTTP 404) and throws a
    /// descriptive <see cref="HttpRequestException"/> on any failure. Never returns mock data.
    /// </summary>
    /// <summary>
    /// Signed version of SendLiveAsync that adds HMAC signatures to requests.
    /// </summary>
    private async Task<T> SendSignedAsync<T>(HttpMethod method, string[] routes, object? body, CancellationToken cancellationToken)
    {
        HttpRequestException? failure = null;
        foreach (var route in routes)
        {
            try
            {
                using var message = CreateSignedRequest(method, route, body);

                using var response = await _http.SendAsync(message, cancellationToken);
                if (response.IsSuccessStatusCode)
                {
                    return await response.Content.ReadFromJsonAsync<T>(JsonOptions, cancellationToken)
                        ?? throw new InvalidOperationException($"ARES API returned an empty response for '{route}'.");
                }

                var detail = await response.Content.ReadAsStringAsync(cancellationToken);
                failure = new HttpRequestException(
                    $"ARES API '{route}' returned {(int)response.StatusCode}: {Truncate(detail)}", null, response.StatusCode);
                if (response.StatusCode != HttpStatusCode.NotFound) break;
            }
            catch (HttpRequestException ex) when (ex.StatusCode is null)
            {
                throw UnreachableError(ex);
            }
        }
        throw failure ?? new HttpRequestException("ARES API request failed.");
    }

    private async Task<T> SendLiveAsync<T>(HttpMethod method, string[] routes, object? body, CancellationToken cancellationToken)
    {
        HttpRequestException? failure = null;
        foreach (var route in routes)
        {
            try
            {
                using var message = new HttpRequestMessage(method, route);
                if (body is not null)
                    message.Content = JsonContent.Create(body, body.GetType(), options: JsonOptions);

                using var response = await _http.SendAsync(message, cancellationToken);
                if (response.IsSuccessStatusCode)
                {
                    return await response.Content.ReadFromJsonAsync<T>(JsonOptions, cancellationToken)
                        ?? throw new InvalidOperationException($"ARES API returned an empty response for '{route}'.");
                }

                var detail = await response.Content.ReadAsStringAsync(cancellationToken);
                failure = new HttpRequestException(
                    $"ARES API '{route}' returned {(int)response.StatusCode}: {Truncate(detail)}", null, response.StatusCode);
                if (response.StatusCode != HttpStatusCode.NotFound) break;
            }
            catch (HttpRequestException ex) when (ex.StatusCode is null)
            {
                throw UnreachableError(ex);
            }
        }
        throw failure ?? new HttpRequestException("ARES API request failed.");
    }

    private HttpRequestException UnreachableError(HttpRequestException inner) =>
        new($"The ARES backend is unreachable at '{_http.BaseAddress}'. Is it running? ({inner.Message})", inner);

    private static string Truncate(string text) => text.Length <= 300 ? text : text[..300] + "...";




    private async Task<T> ReadAsync<T>(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadFromJsonAsync<ApiError>(JsonOptions, cancellationToken);
            throw new HttpRequestException(error?.Message ?? $"ARES API returned {(int)response.StatusCode}.", null, response.StatusCode);
        }
        return await response.Content.ReadFromJsonAsync<T>(JsonOptions, cancellationToken) ?? throw new InvalidOperationException("ARES API returned an empty response.");
    }

    private TestRun ToTestRun(ApiTestRun body)
    {
        var configuration = _submittedConfigurations.GetValueOrDefault(body.Id) ?? ToConfiguration(body.Configuration);
        var finding = body.Findings.FirstOrDefault();
        var evidence = body.Evidence.Select(item => new EvidenceItem(item.Id, item.Source, item.Summary, item.Similarity ?? 0, ParseCategory(item.Category), body.CompletedAt ?? body.CreatedAt)).ToList();
        SecurityClassification? analysis = finding is null ? null : new SecurityClassification(finding.RiskScore, ParseSeverity(finding.Severity), finding.AttackSucceeded, finding.RuntimeClassification, [], evidence, null, null);
        var log = new List<ExecutionLogEvent> { new(body.CreatedAt, "Queued", "Server accepted the controlled test request.") };
        if (body.Status != "queued") log.Add(new(body.CompletedAt ?? DateTimeOffset.UtcNow, ParseStatus(body.Status).ToLabel(), StatusMessage(body), body.Status == "failed" ? "error" : "info"));
        return new TestRun(body.Id, configuration, ParseStatus(body.Status), body.CreatedAt, body.CompletedAt, body.DurationMilliseconds, body.TokenEstimate, analysis, log, body.CorrelationId, body.FailureReason);
    }

    private static ArenaTestConfiguration ToConfiguration(Dictionary<string, JsonElement> configuration) => new()
    {
        TestName = GetString(configuration, "test_name"), TargetApplication = GetString(configuration, "target_application"),
        SystemPrompt = string.Empty, UserPrompt = string.Empty, Provider = ParseProvider(GetString(configuration, "provider")), Model = GetString(configuration, "model"),
        AttackSource = Enum.TryParse<AttackSource>(GetString(configuration, "attack_source"), true, out var source) ? source : AttackSource.Manual,
        AttackCategories = GetCategories(configuration), Temperature = GetDouble(configuration, "temperature"), MaxResponseTokens = GetInt(configuration, "max_response_tokens"), VariationCount = GetInt(configuration, "variation_count"), IncludeHardenedComparison = GetBool(configuration, "include_hardened_comparison")
    };

    private static ArenaTestConfiguration CopyConfiguration(ArenaTestConfiguration source) => new() { TestName = source.TestName, TargetApplication = source.TargetApplication, SystemPrompt = source.SystemPrompt, UserPrompt = source.UserPrompt, Provider = source.Provider, Model = source.Model, AttackSource = source.AttackSource, AttackCategories = [.. source.AttackCategories], Temperature = source.Temperature, MaxResponseTokens = source.MaxResponseTokens, VariationCount = source.VariationCount, IncludeHardenedComparison = source.IncludeHardenedComparison };
    private static string GetString(Dictionary<string, JsonElement> data, string key) => data.TryGetValue(key, out var value) ? value.GetString() ?? string.Empty : string.Empty;
    private static int GetInt(Dictionary<string, JsonElement> data, string key) => data.TryGetValue(key, out var value) && value.TryGetInt32(out var number) ? number : 0;
    private static double GetDouble(Dictionary<string, JsonElement> data, string key) => data.TryGetValue(key, out var value) && value.TryGetDouble(out var number) ? number : 0;
    private static bool GetBool(Dictionary<string, JsonElement> data, string key) => data.TryGetValue(key, out var value) && value.ValueKind == JsonValueKind.True;
    private static List<AttackCategory> GetCategories(Dictionary<string, JsonElement> data, string key = "attack_categories") => data.TryGetValue(key, out var value) && value.ValueKind == JsonValueKind.Array ? value.EnumerateArray().Select(item => ParseCategory(item.GetString() ?? string.Empty)).ToList() : [AttackCategory.DirectPromptInjection];
    private static TestRunStatus ParseStatus(string value) => Enum.Parse<TestRunStatus>(value, true);
    private static AiProvider ParseProvider(string value) => value.Replace("_", "").ToLowerInvariant() switch
    {
        "openai" => AiProvider.OpenAI,
        "groq" => AiProvider.Groq,
        "gemini" => AiProvider.Gemini,
        "nvidianim" or "nvidia" => AiProvider.NvidiaNim,
        "ollama" or "local" => AiProvider.Ollama,
        _ => Enum.TryParse<AiProvider>(value.Replace("_", ""), true, out var prov) ? prov : AiProvider.Groq
    };
    private static AttackCategory ParseCategory(string value) => Enum.Parse<AttackCategory>(value, true);
    private static Severity ParseSeverity(string value) => Enum.Parse<Severity>(value, true);
    private static Challenge ToChallenge(ApiChallenge item) => new(item.Id, item.Title, item.Description, Enum.Parse<ChallengeTrack>(item.Track, true), Enum.Parse<DifficultyTier>(item.Tier, true), ParseCategory(item.Category), item.ScenarioContext, item.Objective, item.Hint, Enum.Parse<DifficultyTier>(item.ModelTier, true), item.TimeParMinutes, item.MaxScore, item.IsRoomLocked, item.Tags);
    private static ChallengeProgressItem ToProgress(ApiProgress item) => new(item.ChallengeId, item.Title, Enum.Parse<ChallengeTrack>(item.Track, true), Enum.Parse<DifficultyTier>(item.Tier, true), Enum.Parse<ChallengeStatus>(item.Status, true), item.BestScore, item.LastAttemptAt);
    private static ChallengeRoom ToRoom(ApiRoom item) => new(item.Id, item.Title, item.Description, item.Theme, item.ChallengeIds, item.PrerequisiteRoomIds, item.BadgeAwardedId);
    private static ChallengeSubmission ToSubmission(ApiSubmission item) => new(item.Id, item.ChallengeId, item.UserId, item.TestRunId, new ScoreBreakdown(item.Score.Total, item.Score.MaxTotal, item.Score.Components.Select(component => new ScoreComponent(component.Label, component.Points, component.MaxPoints, component.Explanation)).ToList(), item.Score.Summary), item.SubmittedAt, item.IsBest);
    private static string StatusMessage(ApiTestRun run) => run.FailureReason ?? (ParseStatus(run.Status) == TestRunStatus.Completed ? "Server-side analysis complete; raw provider output was not retained." : $"Test is {ParseStatus(run.Status).ToLabel().ToLowerInvariant()}.");
    private static string StatusMessage(TestRun run) => run.FailureReason ?? (run.Status == TestRunStatus.Completed ? "Server-side analysis complete; raw provider output was not retained." : $"Test is {run.Status.ToLabel().ToLowerInvariant()}.");

    private sealed record ApiCreateTestResponse(string TestId, string Status, string CorrelationId, DateTimeOffset CreatedAt);
    private sealed record ApiCancelTestResponse(string TestId, string Status, string CorrelationId);
    private sealed record ApiTestRun(string Id, string Status, DateTimeOffset CreatedAt, DateTimeOffset? CompletedAt, int DurationMilliseconds, int TokenEstimate, string CorrelationId, string? FailureReason, Dictionary<string, JsonElement> Configuration, List<ApiFinding> Findings, List<ApiEvidence> Evidence);
    private sealed record ApiFinding(string Id, string Category, string Severity, int RiskScore, bool AttackSucceeded, string RuntimeClassification, string Summary);
    private sealed record ApiEvidence(string Id, string Source, string Category, string Summary, double? Similarity, bool Redacted);
    private sealed record ApiRecentTests(List<ApiRecentTest> Items);
    private sealed record ApiRecentTest(string Id, DateTimeOffset Timestamp, string Category, string Provider, string Model, int RiskScore, string Status);
    private sealed record ApiDashboardSummary(List<ApiMetric> Metrics, int CorpusSize, int ProtectedApplications, DateTimeOffset GeneratedAt, bool IsPartial);
    private sealed record ApiMetric(string Label, string Value, string Change, string Trend, string Status, string Description);
    private sealed record ApiTrends(List<ApiTrend> Points);
    private sealed record ApiTrend(DateOnly Date, int Tested, int Blocked, int Successful, int Incidents);
    private sealed record ApiCategories(List<ApiCategory> Items);
    private sealed record ApiCategory(string Category, int Tests, int Successful, int Blocked, int SuccessRate);
    private sealed record ApiIncidents(List<ApiIncident> Items);
    private sealed record ApiIncident(string Id, string Application, string Category, string Severity, DateTimeOffset DetectedAt, string EnforcementAction, string Status, string CorrelationId);
    private sealed record ApiHardening(int BaselineSuccessRate, int HardenedSuccessRate, int ImprovementPoints, int TestsIncluded, DateOnly? LastCycle);
    private sealed record ApiProviders(List<ApiProvider> Items);
    private sealed record ApiProvider(string? Provider, string Name, string Status, string Detail, DateTimeOffset CheckedAt);
    private sealed record ApiModels(List<ApiModel> Items);
    private sealed record ApiModel(string Provider, string Id, string DisplayName, int ContextWindow, bool IsAvailable);
    private sealed record ApiChallengePage(List<ApiChallenge> Items, List<ApiProgress> Progress, int TotalCount, int Page, int PageSize);
    private sealed record ApiChallenge(string Id, string Title, string Description, string Track, string Tier, string Category, string ScenarioContext, string Objective, string? Hint, string ModelTier, int TimeParMinutes, int MaxScore, bool IsRoomLocked, List<string> Tags);
    private sealed record ApiProgress(string ChallengeId, string Title, string Track, string Tier, string Status, int? BestScore, DateTimeOffset? LastAttemptAt);
    private sealed record ApiRoom(string Id, string Title, string Description, string Theme, List<string> ChallengeIds, List<string> PrerequisiteRoomIds, string BadgeAwardedId);
    private sealed record ApiRooms(List<ApiRoom> Items);
    private sealed record ApiPaths(List<ApiPath> Items);
    private sealed record ApiPath(string Id, string Name, string Description, List<string> RoomIds, string BadgeAwarded, string Colour);
    private sealed record ApiScore(int Total, int MaxTotal, List<ApiScoreComponent> Components, string Summary);
    private sealed record ApiScoreComponent(string Label, double Points, double MaxPoints, string Explanation);
    private sealed record ApiSubmission(string Id, string ChallengeId, string UserId, string TestRunId, ApiScore Score, DateTimeOffset SubmittedAt, bool IsBest);
    private sealed record ApiSubmitChallenge(ApiSubmission Submission, bool BadgeUnlocked, string? NextChallengeId);
    private sealed record ApiSubmissions(List<ApiSubmission> Items);
    private sealed record ApiProgressList(List<ApiProgress> Items);
    private sealed record ApiProfile(string Id, string DisplayName, string Initials, int XpTotal, int Level, int XpThisLevel, int XpToNextLevel, int ChallengesSolved, int AttackerSolved, int DefenderSolved, DateTimeOffset MemberSince);
}
