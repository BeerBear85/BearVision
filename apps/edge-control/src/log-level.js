const LOG_LEVEL_RANK = {
  debug: 10,
  info: 20,
  warning: 30,
  error: 40,
};

export function runtimeLogLevel(event) {
  if (event?.kind !== "runtime_log") return null;

  const match = String(event.payload?.message ?? "").match(/\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b/i);
  const level = match?.[1]?.toLowerCase();
  if (level === "critical") return "error";
  return level && level in LOG_LEVEL_RANK ? level : "info";
}

export function showsAtMinimumLogLevel(event, minimumLevel) {
  const eventLevel = runtimeLogLevel(event);
  if (eventLevel === null) return true;

  const minimumRank = LOG_LEVEL_RANK[minimumLevel] ?? LOG_LEVEL_RANK.info;
  return LOG_LEVEL_RANK[eventLevel] >= minimumRank;
}

function traceBucket(event) {
  return runtimeLogLevel(event) ?? "event";
}

export function appendRetainedTraceEvent(current, event, maxPerBucket = 200) {
  const retained = [];
  const bucketCounts = new Map();

  for (const candidate of [event, ...current]) {
    const bucket = traceBucket(candidate);
    const count = bucketCounts.get(bucket) ?? 0;
    if (count >= maxPerBucket) continue;
    retained.push(candidate);
    bucketCounts.set(bucket, count + 1);
  }

  return retained;
}
