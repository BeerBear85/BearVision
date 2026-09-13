import assert from "node:assert/strict";
import test from "node:test";

import { withAdminBasicAuth } from "./admin-basic-auth.mjs";

const validEnvironment = {
  BEARVISION_ADMIN_USERNAME: "admin",
  BEARVISION_ADMIN_PASSWORD: "correct horse battery staple",
};

function recordingResponse() {
  return {
    status: undefined,
    headers: undefined,
    body: undefined,
    writeHead(status, headers) {
      this.status = status;
      this.headers = headers;
    },
    end(body) { this.body = body; },
  };
}

test("admin authentication construction fails closed for invalid credentials", () => {
  const invalidEnvironments = [
    {},
    { BEARVISION_ADMIN_USERNAME: "admin" },
    { BEARVISION_ADMIN_PASSWORD: "password" },
    { ...validEnvironment, BEARVISION_ADMIN_USERNAME: "" },
    { ...validEnvironment, BEARVISION_ADMIN_PASSWORD: "" },
    { ...validEnvironment, BEARVISION_ADMIN_USERNAME: "admin:user" },
    { ...validEnvironment, BEARVISION_ADMIN_USERNAME: "admin\u0000" },
    { ...validEnvironment, BEARVISION_ADMIN_USERNAME: "admin\u007f" },
    { ...validEnvironment, BEARVISION_ADMIN_PASSWORD: "line\rbreak" },
    { ...validEnvironment, BEARVISION_ADMIN_PASSWORD: "line\nbreak" },
  ];

  for (const environment of invalidEnvironments) {
    assert.throws(
      () => withAdminBasicAuth(() => {}, environment),
      { message: "invalid Server Control authentication configuration" },
    );
  }
});

test("valid UTF-8 credentials with a colon in the password reach the handler once", () => {
  const environment = {
    BEARVISION_ADMIN_USERNAME: "bjørn",
    BEARVISION_ADMIN_PASSWORD: "løsen:ord",
  };
  const encoded = Buffer.from("bjørn:løsen:ord", "utf8").toString("base64");
  const authorization = `bAsIc  ${encoded}`;
  let calls = 0;
  const request = {
    headers: { authorization },
    rawHeaders: ["Authorization", authorization],
  };

  withAdminBasicAuth(() => { calls += 1; }, environment)(request, {});

  assert.equal(calls, 1);
});

test("a missing Authorization header gets the generic authentication challenge", () => {
  let calls = 0;
  const response = recordingResponse();

  withAdminBasicAuth(() => { calls += 1; }, validEnvironment)(
    { headers: {}, rawHeaders: [] },
    response,
  );

  assert.equal(calls, 0);
  assert.equal(response.status, 401);
  assert.deepEqual(response.headers, {
    "WWW-Authenticate": 'Basic realm="BearVision Server Control", charset="UTF-8"',
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
  });
  assert.equal(response.body, '{"error":"authentication required"}');
});

test("wrong, malformed, non-Basic, and duplicate headers get the same rejection", () => {
  const valid = `Basic ${Buffer.from("admin:correct horse battery staple").toString("base64")}`;
  const cases = [
    ["wrong username", `Basic ${Buffer.from("other:correct horse battery staple").toString("base64")}`, validEnvironment],
    ["wrong password", `Basic ${Buffer.from("admin:wrong").toString("base64")}`, validEnvironment],
    ["malformed base64", "Basic !!!", validEnvironment],
    ["invalid UTF-8", "Basic /zo=", validEnvironment],
    ["missing separator", `Basic ${Buffer.from("admin-password").toString("base64")}`, validEnvironment],
    ["wrong scheme", `Bearer ${valid.slice(6)}`, validEnvironment],
    ["partial base64 padding", "Basic YWRtaW46cGFzcw=", {
      BEARVISION_ADMIN_USERNAME: "admin", BEARVISION_ADMIN_PASSWORD: "pass",
    }],
  ];

  for (const [label, authorization, environment] of cases) {
    let calls = 0;
    const response = recordingResponse();
    withAdminBasicAuth(() => { calls += 1; }, environment)(
      { headers: { authorization }, rawHeaders: ["Authorization", authorization] },
      response,
    );
    assert.equal(calls, 0, label);
    assert.equal(response.status, 401, label);
    assert.deepEqual(response.headers, {
      "WWW-Authenticate": 'Basic realm="BearVision Server Control", charset="UTF-8"',
      "Cache-Control": "no-store",
      "Content-Type": "application/json; charset=utf-8",
    }, label);
    assert.equal(response.body, '{"error":"authentication required"}', label);
  }

  let duplicateCalls = 0;
  const duplicateResponse = recordingResponse();
  withAdminBasicAuth(() => { duplicateCalls += 1; }, validEnvironment)(
    {
      headers: { authorization: valid },
      rawHeaders: ["Authorization", valid, "authorization", valid],
    },
    duplicateResponse,
  );
  assert.equal(duplicateCalls, 0);
  assert.equal(duplicateResponse.status, 401);
  assert.equal(duplicateResponse.headers["Cache-Control"], "no-store");
  assert.equal(duplicateResponse.body, '{"error":"authentication required"}');
});
