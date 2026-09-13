import { createHash, timingSafeEqual } from "node:crypto";

const CONFIGURATION_ERROR = "invalid Server Control authentication configuration";
const USERNAME_CONTROL_CHARACTER = /[\u0000-\u001f\u007f-\u009f]/u;
const AUTHENTICATION_BODY = '{"error":"authentication required"}';
const AUTHENTICATION_CHALLENGE = 'Basic realm="BearVision Server Control", charset="UTF-8"';
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: true });

function credentialsFromEnvironment(env) {
  const username = env?.BEARVISION_ADMIN_USERNAME;
  const password = env?.BEARVISION_ADMIN_PASSWORD;
  if (
    typeof username !== "string"
    || typeof password !== "string"
    || username.length === 0
    || password.length === 0
    || username.includes(":")
    || USERNAME_CONTROL_CHARACTER.test(username)
    || /[\r\n]/u.test(password)
  ) {
    throw new Error(CONFIGURATION_ERROR);
  }
  return { username, password };
}

function digest(value) {
  return createHash("sha256").update(value, "utf8").digest();
}

function authorizationHeaders(request) {
  if (Array.isArray(request.rawHeaders)) {
    const values = [];
    for (let index = 0; index < request.rawHeaders.length; index += 2) {
      if (String(request.rawHeaders[index]).toLowerCase() === "authorization") {
        values.push(request.rawHeaders[index + 1]);
      }
    }
    return values;
  }
  const distinct = request.headersDistinct?.authorization;
  if (Array.isArray(distinct)) return distinct;
  const value = request.headers?.authorization;
  if (Array.isArray(value)) return value;
  return typeof value === "string" ? [value] : [];
}

function parseBasicCredentials(header) {
  if (typeof header !== "string") return null;
  const match = /^Basic +([A-Za-z0-9+/]+={0,2})$/iu.exec(header);
  if (!match) return null;
  const token = match[1];
  const hasPadding = token.includes("=");
  if ((hasPadding && token.length % 4 !== 0) || (!hasPadding && token.length % 4 === 1)) {
    return null;
  }
  const padded = token + "=".repeat((4 - (token.length % 4)) % 4);
  const bytes = Buffer.from(padded, "base64");
  if (bytes.toString("base64") !== padded) return null;
  let decoded;
  try {
    decoded = UTF8_DECODER.decode(bytes);
  } catch {
    return null;
  }
  const separator = decoded.indexOf(":");
  if (separator < 0) return null;
  return { username: decoded.slice(0, separator), password: decoded.slice(separator + 1) };
}

function rejectAuthentication(response) {
  response.writeHead(401, {
    "WWW-Authenticate": AUTHENTICATION_CHALLENGE,
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
  });
  response.end(AUTHENTICATION_BODY);
}

export function withAdminBasicAuth(handler, env) {
  const expected = credentialsFromEnvironment(env);
  const expectedUsername = digest(expected.username);
  const expectedPassword = digest(expected.password);
  return (request, response) => {
    const headers = authorizationHeaders(request);
    const credentials = headers.length === 1 ? parseBasicCredentials(headers[0]) : null;
    if (credentials) {
      const usernameMatches = timingSafeEqual(digest(credentials.username), expectedUsername);
      const passwordMatches = timingSafeEqual(digest(credentials.password), expectedPassword);
      if (usernameMatches && passwordMatches) return handler(request, response);
    }
    rejectAuthentication(response);
  };
}
