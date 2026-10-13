using Ares.Web.Models;

namespace Ares.Web.Services;

/// <summary>Contract for the ARES API client — implemented by FastApiAresApiClient (live) and MockAresApiClient (dev).</summary>
public interface IAresApiClient
{
    Task<CreateTestResponse> CreateTestAsync(CreateTestRequest request, CancellationToken cancellationToken);
    Task<TestRun?> GetTestAsync(string testId, CancellationToken cancellationToken);
    Task<TestRun> SimulateTestAsync(string testId, IProgress<ExecutionProgress>? progress, CancellationToken cancellationToken);
    Task<CancelTestResponse> CancelTestAsync(string testId, CancellationToken cancellationToken);
    Task<IReadOnlyList<RecentTestItem>> GetRecentTestsAsync(CancellationToken cancellationToken);
    Task<CorpusSaveResponse> SaveAttackToCorpusAsync(CorpusSaveRequest request, CancellationToken cancellationToken);
    Task<DashboardSummary> GetDashboardSummaryAsync(CancellationToken cancellationToken);
    Task<IReadOnlyList<TrendPoint>> GetDashboardTrendsAsync(CancellationToken cancellationToken);
    Task<IReadOnlyList<CategoryMetric>> GetCategoryMetricsAsync(CancellationToken cancellationToken);
    Task<IReadOnlyList<RuntimeIncident>> GetRecentIncidentsAsync(CancellationToken cancellationToken);
    Task<HardeningComparison> GetHardeningComparisonAsync(CancellationToken cancellationToken);
    Task<IReadOnlyList<ProviderHealth>> GetProviderHealthAsync(CancellationToken cancellationToken);
    Task<IReadOnlyList<ModelConfiguration>> GetModelsAsync(AiProvider provider, CancellationToken cancellationToken);

    // Phase 6 — ML Stats, Semantic Search, and Prompt Hardening
    Task<StatsResponse> GetStatsAsync(CancellationToken cancellationToken);
    Task<StatsResponse> RetrainStatsAsync(CancellationToken cancellationToken);
    Task<string> GetReportAsync(string testId, string format, CancellationToken cancellationToken);
    Task<SearchResponse> SearchAsync(SearchRequest request, CancellationToken cancellationToken);
    Task<HardenResponse> HardenAsync(HardenRequest request, CancellationToken cancellationToken);

    // ── Arena Platform ────────────────────────────────────────────────────────

    // Challenge catalogue
    Task<ChallengePageResult> GetChallengesAsync(ChallengeTrack? track, DifficultyTier? tier, AttackCategory? category, string? search, int page, CancellationToken cancellationToken);
    Task<Challenge?> GetChallengeAsync(string challengeId, CancellationToken cancellationToken);
    Task<IReadOnlyList<ChallengeRoom>> GetRoomsAsync(CancellationToken cancellationToken);
    Task<ChallengeRoom?> GetRoomAsync(string roomId, CancellationToken cancellationToken);
    Task<IReadOnlyList<LearningPath>> GetLearningPathsAsync(CancellationToken cancellationToken);

    // Submission & scoring
    Task<SubmitChallengeResponse> SubmitChallengeAsync(SubmitChallengeRequest request, CancellationToken cancellationToken);
    Task<IReadOnlyList<ChallengeSubmission>> GetMySubmissionsAsync(string challengeId, CancellationToken cancellationToken);
    Task<IReadOnlyList<ChallengeProgressItem>> GetMyProgressAsync(CancellationToken cancellationToken);

    // Profile
    Task<UserProfile> GetMyProfileAsync(CancellationToken cancellationToken);

    // Org / Admin
    Task<IReadOnlyList<OrgAssessment>> GetOrgAssessmentsAsync(CancellationToken cancellationToken);
    Task<OrgAssessment> CreateOrgAssessmentAsync(CreateOrgAssessmentRequest request, CancellationToken cancellationToken);
    Task<Challenge> CreateChallengeAsync(CreateChallengeRequest request, CancellationToken cancellationToken);
    Task<Challenge> UpdateChallengeAsync(UpdateChallengeRequest request, CancellationToken cancellationToken);

    // ── Runtime Enforcement Gateway ──────────────────────────────────────────
    Task<GatewayEnforceResponse> EnforcePromptAsync(GatewayEnforceRequest request, CancellationToken cancellationToken);
    Task<GatewayStatusResponse> GetGatewayStatusAsync(CancellationToken cancellationToken);

    // ── User Prompt Optimizer ────────────────────────────────────────────────
    Task<OptimizePromptResponse> OptimizePromptAsync(OptimizePromptRequest request, CancellationToken cancellationToken);
    Task<IReadOnlyList<IntentInfoResponse>> GetIntentCategoriesAsync(CancellationToken cancellationToken);
}

