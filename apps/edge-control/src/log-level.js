const LOG_LEVEL_RANK = {
  debug: 10,
  info: 20,
  warning: 30,
  error: 40,
};

function normalizedLevel(value) {
  const level = String(value ?? "").toLowerCase();
  if (level === "critical") return "error";
  return level in LOG_LEVEL_RANK ? level : null;
}

function levelDeclaredByLine(message) {
  const text = String(message ?? "");
  const header = text.match(/\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+([\w.]+):\s*(.*)$/i);
  if (header) {
    const level = normalizedLevel(header[1]);
    const loggerName = header[2].toLowerCase();
    const body = header[3];
    if (level === "info" && loggerName.startsWith("open_gopro.") && body === "") {
      return "debug";
    }
    return level;
  }

  const explicitLevel = text.match(/\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b/i)?.[1];
  return normalizedLevel(explicitLevel);
}

export function createRuntimeLogLevelClassifier(defaultLevel = "info") {
  let inheritedLevel = normalizedLevel(defaultLevel) ?? "info";

  return (message) => {
    const declaredLevel = levelDeclaredByLine(message);
    if (declaredLevel) inheritedLevel = declaredLevel;
    return inheritedLevel;
  };
}

export function runtimeLogLevel(event) {
  if (event?.kind !== "runtime_log") return null;

  return normalizedLevel(event.payload?.level)
    ?? levelDeclaredByLine(event.payload?.message)
    ?? "info";
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
