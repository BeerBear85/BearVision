function encodeSse(id, event) {
  return `id: ${id}\ndata: ${JSON.stringify(event)}\n\n`;
}

export class EventStream {
  constructor({
    replayLimit = 500,
    heartbeatMs = 15_000,
    getSnapshot = () => ({}),
    setTimer = setTimeout,
    clearTimer = clearTimeout,
  } = {}) {
    this.replayLimit = replayLimit;
    this.heartbeatMs = heartbeatMs;
    this.getSnapshot = getSnapshot;
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
    this.sequence = 0;
    this.history = [];
    this.clients = new Set();
    this.heartbeatTimer = null;
  }

  publish(event) {
    const item = { id: ++this.sequence, event: structuredClone(event) };
    this.history = [...this.history, item].slice(-this.replayLimit);
    const message = encodeSse(item.id, item.event);
    for (const response of this.clients) response.write(message);
    return item.id;
  }

  connect(request, response) {
    response.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    const rawLastId = request.headers?.["last-event-id"];
    const lastId = rawLastId == null || rawLastId === "" ? null : Number(rawLastId);
    const oldestId = this.history[0]?.id ?? this.sequence + 1;
    const stale = Number.isFinite(lastId)
      && (lastId < oldestId - 1 || lastId > this.sequence);

    if (lastId == null || !Number.isFinite(lastId) || stale) {
      response.write(encodeSse(this.sequence, {
        kind: "control_snapshot",
        payload: this.getSnapshot(),
      }));
    } else {
      for (const item of this.history) {
        if (item.id > lastId) response.write(encodeSse(item.id, item.event));
      }
    }

    this.clients.add(response);
    request.on("close", () => {
      this.clients.delete(response);
      if (this.clients.size === 0) this.#stopHeartbeat();
    });
    this.#startHeartbeat();
  }

  #startHeartbeat() {
    if (this.heartbeatTimer != null || this.clients.size === 0) return;
    this.heartbeatTimer = this.setTimer(() => {
      this.heartbeatTimer = null;
      for (const response of this.clients) response.write(": heartbeat\n\n");
      this.#startHeartbeat();
    }, this.heartbeatMs);
  }

  #stopHeartbeat() {
    if (this.heartbeatTimer == null) return;
    this.clearTimer(this.heartbeatTimer);
    this.heartbeatTimer = null;
  }

  close() {
    this.#stopHeartbeat();
    for (const response of this.clients) response.end();
    this.clients.clear();
  }
}
