export function parseRuntimeEventLine(line) {
  const event = JSON.parse(line);
  if (event.control_event_version !== "1.0") {
    throw new Error("unsupported runtime event version");
  }
  if (typeof event.kind !== "string" || !event.kind) {
    throw new Error("runtime event kind is required");
  }
  if (event.at_s !== null && (!Number.isFinite(event.at_s) || event.at_s < 0)) {
    throw new Error("runtime event time is invalid");
  }
  if (!event.payload || typeof event.payload !== "object" || Array.isArray(event.payload)) {
    throw new Error("runtime event payload must be an object");
  }
  return event;
}
