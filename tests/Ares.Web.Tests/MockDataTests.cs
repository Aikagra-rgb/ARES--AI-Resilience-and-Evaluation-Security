using Ares.Web.Models;
using Ares.Web.Services;
using Xunit;

namespace Ares.Web.Tests;

public sealed class MockDataTests
{
    [Fact]
    public void Recent_tests_cover_required_terminal_states()
    {
        var statuses = MockData.RecentTestRuns().Select(run => run.Status).ToHashSet();

        Assert.Contains(TestRunStatus.Completed, statuses);
        Assert.Contains(TestRunStatus.Blocked, statuses);
        Assert.Contains(TestRunStatus.Failed, statuses);
        Assert.Contains(TestRunStatus.Cancelled, statuses);
    }

    [Fact]
    public void Direct_injection_produces_critical_success_with_comparison()
    {
        var configuration = new ArenaTestConfiguration { AttackCategories = [AttackCategory.DirectPromptInjection] };

        var result = MockData.CreateAnalysis(configuration);

        Assert.NotNull(result.Analysis);
        Assert.Equal(Severity.Critical, result.Analysis!.Severity);
        Assert.True(result.Analysis.AttackSucceeded);
        Assert.NotNull(result.Analysis.Comparison);
    }

    [Fact]
    public void NIM_tool_misuse_is_a_safe_provider_failure()
    {
        var configuration = new ArenaTestConfiguration { Provider = AiProvider.NvidiaNim, Model = "meta/llama-3.1-70b-instruct", AttackCategories = [AttackCategory.ToolMisuse] };

        var result = MockData.CreateAnalysis(configuration);

        Assert.Null(result.Analysis);
        Assert.NotNull(result.FailureReason);
        Assert.DoesNotContain("key", result.FailureReason!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Display_labels_are_human_readable()
    {
        Assert.Equal("Direct prompt injection", AttackCategory.DirectPromptInjection.ToLabel());
        Assert.Equal("NVIDIA NIM", AiProvider.NvidiaNim.ToLabel());
    }

    [Theory]
    [InlineData(AttackCategory.DirectPromptInjection, AiProvider.OpenAI, "gpt-4.1-mini", TestRunStatus.Completed)]
    [InlineData(AttackCategory.DataExfiltration, AiProvider.Gemini, "gemini-2.5-flash", TestRunStatus.Blocked)]
    [InlineData(AttackCategory.ToolMisuse, AiProvider.NvidiaNim, "meta/llama-3.1-70b-instruct", TestRunStatus.Failed)]
    public async Task Simulated_run_reaches_expected_terminal_state(AttackCategory category, AiProvider provider, string model, TestRunStatus expected)
    {
        var client = new MockAresApiClient();
        var configuration = new ArenaTestConfiguration { Provider = provider, Model = model, AttackCategories = [category] };
        var created = await client.CreateTestAsync(new CreateTestRequest(configuration), CancellationToken.None);
        var result = await client.SimulateTestAsync(created.TestId, null, CancellationToken.None);

        Assert.Equal(expected, result.Status);
        Assert.Contains(result.Log, item => item.Stage is "Completed" or "Provider failure");
    }

    [Fact]
    public async Task Cancelled_run_is_persisted_as_cancelled()
    {
        var client = new MockAresApiClient();
        var created = await client.CreateTestAsync(new CreateTestRequest(new ArenaTestConfiguration()), CancellationToken.None);

        var cancellation = await client.CancelTestAsync(created.TestId, CancellationToken.None);
        var result = await client.GetTestAsync(created.TestId, CancellationToken.None);

        Assert.Equal(TestRunStatus.Cancelled, cancellation.Status);
        Assert.Equal(TestRunStatus.Cancelled, result!.Status);
    }

    [Fact]
    public async Task Challenge_submission_updates_only_the_private_profile_progress()
    {
        var client = new MockAresApiClient();
        var before = await client.GetMyProfileAsync(CancellationToken.None);

        var submission = await client.SubmitChallengeAsync(new SubmitChallengeRequest("CHK-A-001", "TST-260820-100"), CancellationToken.None);
        var after = await client.GetMyProfileAsync(CancellationToken.None);
        var progress = await client.GetMyProgressAsync(CancellationToken.None);

        Assert.True(submission.Submission.Score.Total > 0);
        Assert.Equal(before.XpTotal + submission.Submission.Score.Total, after.XpTotal);
        Assert.Equal(before.AttackerSolved + 1, after.AttackerSolved);
        Assert.Contains(progress, item => item.ChallengeId == "CHK-A-001" && item.Status == ChallengeStatus.Completed);
    }

    [Fact]
    public async Task Rooms_remain_optional_collections()
    {
        var client = new MockAresApiClient();

        var rooms = await client.GetRoomsAsync(CancellationToken.None);
        var catalogue = await client.GetChallengesAsync(null, null, null, null, 1, CancellationToken.None);

        Assert.NotEmpty(rooms);
        Assert.NotEmpty(catalogue.Items);
        Assert.Contains(rooms, room => room.PrerequisiteRoomIds.Count == 0);
    }

    [Fact]
    public async Task OptimizePrompt_populates_calibrated_fidelity_verification_and_mitigated_threats()
    {
        var client = new MockAresApiClient();

        var directiveResult = await client.OptimizePromptAsync(
            new OptimizePromptRequest("Set operating policy: strictly enforce that agents never share customer IDs.", "general", null, false),
            CancellationToken.None
        );

        Assert.Equal("OPTIMIZED", directiveResult.Status);
        Assert.Equal("SYSTEM_DIRECTIVE", directiveResult.Intent);
        Assert.Equal(0.80, directiveResult.TargetFidelity);
        Assert.True(directiveResult.VerificationPassed);
        Assert.NotNull(directiveResult.MitigatedThreats);
        Assert.NotEmpty(directiveResult.MitigatedThreats!);
    }
}
