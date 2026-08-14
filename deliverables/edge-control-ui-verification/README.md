# Edge Control UI verification

Verified against the production build served from `apps/edge-control/dist`.

## Screenshots

- `edge-control-desktop.png`: 1440 × 1000 desktop viewport.
- `edge-control-mobile.png`: 390 × 844 mobile viewport.

## Browser checks

Desktop:

- no horizontal page overflow;
- all buttons, links and the scenario selector remain inside the viewport;
- preview and status rail are visible;
- no browser console warnings or errors.

Mobile:

- no horizontal page overflow;
- all buttons, links and the scenario selector remain inside the viewport;
- the dashboard and status rail resolve to one column;
- component source labels wrap without an internal scrollbar.

Interaction smoke test:

- selecting Hardware hides the scenario selector and changes the primary action
  to `Start hardware`;
- selecting Simulation restores the scenario selector and `Run scenario`;
- `Stop runtime` remains disabled while the runtime is idle.

## Automated checks

- Vite production build passes.
- All eight Edge Control Node tests pass.

This is a browser-driven verification record, not a committed pixel-diff
baseline. The repository does not yet have a React end-to-end test runner such
as Playwright configured in CI.
