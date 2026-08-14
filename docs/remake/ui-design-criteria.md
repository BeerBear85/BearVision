# BearVision operator UI design criteria

Status: shared baseline for Edge Control and Server Control.

## Product intent

The two control applications are one BearVision product family, but they serve
different jobs. Server Control supports administration and review. Edge Control
supports a time-sensitive operator flow: configure, start, observe, stop and
verify. Shared styling must not erase that distinction.

## Required criteria

1. **Make the current state unmistakable.** The runtime, connection and capture
   states must be visible without opening technical details. Status uses text
   and shape in addition to colour.
2. **Keep one primary action per state.** Starting is primary while idle;
   stopping is available only while active. Configuration is locked during a
   run so the displayed setup always matches the active runtime.
3. **Follow the operator's sequence.** Setup precedes preview; live health and
   activity support observation. Output switching appears only when artefacts
   exist.
4. **Use progressive disclosure.** The preview and operational state dominate.
   Component sources, tracking overlays and the full event trace remain
   available as evidence without competing with the start/stop flow.
5. **Share the BearVision visual language.** Both applications use the same
   green sidebar, amber BV mark, light work surfaces, typography, form shapes,
   focus treatment and semantic status colours.
6. **Design for the deployed screen and smaller fallbacks.** The full workspace
   is optimised for a desktop Edge computer. It must still reflow at 860 px and
   620 px without horizontal page scrolling or hidden controls.
7. **Meet keyboard and assistive-technology basics.** Controls have visible
   focus, selected states use `aria-pressed`, failures use `role="alert"`, live
   state changes use restrained live regions, and headings describe page
   structure.
8. **Preserve technical truth.** The UI must not imply that the Edge application
   identifies the rider, uploads independently or provides a hardware preview
   before those capabilities exist. Labels must reflect ownership and runtime
   state from the control API.

## Edge Control acceptance checks

- An operator can identify mode, runtime phase and connection state at a glance.
- Simulation selection and hardware selection remain mutually exclusive.
- Scenario choice cannot change during a run.
- Source, extracted, processed and tracking media remain reachable when present.
- GoPro capture and the event trace remain visible. Rider assignment stays in
  Server Control, which owns that decision.
- The page remains operable at 320 px without a fixed desktop minimum width.
- The production build and Node control tests pass.

## Deliberate non-goals

- Visual parity does not mean identical navigation or content density.
- Edge Control does not duplicate Server Control's users, jobs or video library.
- Colour alone never communicates success, activity or failure.
- New controls are not added for backend capabilities that do not yet exist.
