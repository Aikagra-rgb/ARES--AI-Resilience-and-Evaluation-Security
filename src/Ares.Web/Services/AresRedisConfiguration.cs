namespace Ares.Web.Services;

/// <summary>
/// Normalizes Redis connection strings (supporting both `redis://`/`rediss://` URLs
/// and native StackExchange.Redis `host:port,password=...` strings) and enforces
/// multi-instance session requirements outside local development.
/// </summary>
public static class AresRedisConfiguration
{
    public static string NormalizeConnectionString(string raw)
    {
        var trimmed = (raw ?? string.Empty).Trim();
        if (string.IsNullOrWhiteSpace(trimmed))
        {
            return string.Empty;
        }

        if (!trimmed.StartsWith("redis://", StringComparison.OrdinalIgnoreCase) &&
            !trimmed.StartsWith("rediss://", StringComparison.OrdinalIgnoreCase))
        {
            return trimmed;
        }

        var uri = new Uri(trimmed);
        var host = uri.Host;
        var port = uri.Port > 0 ? uri.Port : (uri.Scheme.Equals("rediss", StringComparison.OrdinalIgnoreCase) ? 6380 : 6379);
        var parts = new List<string> { $"{host}:{port}" };

        if (!string.IsNullOrEmpty(uri.UserInfo))
        {
            var userInfo = uri.UserInfo;
            var colonIdx = userInfo.IndexOf(':');
            if (colonIdx >= 0)
            {
                var user = Uri.UnescapeDataString(userInfo[..colonIdx]);
                var pass = Uri.UnescapeDataString(userInfo[(colonIdx + 1)..]);
                if (!string.IsNullOrEmpty(user))
                {
                    parts.Add($"user={user}");
                }
                if (!string.IsNullOrEmpty(pass))
                {
                    parts.Add($"password={pass}");
                }
            }
            else
            {
                parts.Add($"password={Uri.UnescapeDataString(userInfo)}");
            }
        }

        if (uri.Scheme.Equals("rediss", StringComparison.OrdinalIgnoreCase))
        {
            parts.Add("ssl=true");
        }

        var dbSegment = uri.AbsolutePath.Trim('/');
        if (!string.IsNullOrEmpty(dbSegment) && int.TryParse(dbSegment, out var dbIndex) && dbIndex >= 0)
        {
            parts.Add($"defaultDatabase={dbIndex}");
        }

        return string.Join(",", parts);
    }

    public static void ValidateSessionStorageConfiguration(
        bool isDevelopment,
        string? redisConnection,
        bool allowSingleNodeSessions)
    {
        if (!isDevelopment && string.IsNullOrWhiteSpace(redisConnection) && !allowSingleNodeSessions)
        {
            throw new InvalidOperationException(
                "Multi-instance Redis session storage (Ares:RedisConnection or REDIS_URL) is required in Production. " +
                "Set Ares:AllowSingleNodeSessions=true only for explicit single-node deployments."
            );
        }
    }
}
