# Server Control exploratory operatørtest — 11. september 2026

## Dokumentstatus

Dette dokument er den versionerede baseline for den gennemførte test af commit
`bfcc7a6e1fd3db2d71e1b68d357e9e4e4a0c8561`. Nye testkørsler skal oprette en
ny dateret rapport og henvise til denne baseline frem for at overskrive de
oprindelige observationer.

Screenshots, testharnesses og captures under
`temp/server-control-operator-20260911/` er lokale testartefakter, som er
udelukket fra Git. Evidenslinkene virker derfor kun i det oprindelige workspace;
tabellen og de beskrevne observationer udgør den versionerede dokumentation.

## Konklusion

Deploymenten lykkedes lokalt, og de basale flows for modtagelse, automatisk
assignment, unresolved-triage og requeue fungerer. Server Control er dog ikke
operatørklar til korrektion af identitetsdata: manuel omfordeling af klip,
redigering af brugerdata, ændring af eksisterende BearTag-intervaller,
konsekvensanalyse og genberegning af allerede processerede assignments mangler
i UI'en.

Samlet vurdering: **egnet til overvågning og simpel triage, ikke egnet som
autoritativt administrationsværktøj uden udviklerhjælp**.

## Deployment

- Status: kører lokalt på `http://127.0.0.1:4320`.
- Branch: `master` mod `origin/master`.
- Commit: `bfcc7a6e1fd3db2d71e1b68d357e9e4e4a0c8561`.
- Konfiguration: `config/server.local.yaml`.
- Data: lokal registry `data/server/user-registry.json`, lokal kø
  `temp/simulation-queue` og lokal scratch `temp/server-scratch`.
- Produktionsdata/Box-credentials: ikke anvendt.
- Dependencies: `uv sync --locked --extra dev` og
  `corepack pnpm install --frozen-lockfile` gennemført.
- Tests: 10/10 Node-tests og 25/25 relevante Python-tests bestod. Første
  Python-kørsel kunne ikke læse Windows' globale pytest-temp; samme suite bestod
  med en ny `--basetemp` under repositoryet.
- Build: Vite production build bestod.
- UI-driver: den native Windows-driver fejlede under initialisering. Testen blev
  derfor udført mod den synlige browser-UI med repositoryets Playwright-runtime
  og en allerede installeret Microsoft Edge. Screenshots og UI-tekst er stadig
  observeret gennem den renderede UI, ikke vurderet ud fra kildekoden.

## Testdata

Ved start fandtes to lokale unresolved testklip og ingen users/BearTags.

Oprettet gennem UI:

- `Operator Test Rider Alpha 20260911`,
  `alpha.operator.20260911@example.com`, med `tag-17` fra
  31-12-2025 00:00 til 01-01-2027 00:00.
- `Operator Test Rider Bravo 20260911`,
  `bravo.operator.20260911@example.com`, med
  `operator-test-tag-b-20260911` i samme interval.
- En ekstra Alpha-fixture med `example.test` blev oprettet under det første
  submit-forsøg, fordi formularen blev stående uden gemt/progress-feedback,
  mens read model-opdateringen var langsom. Den er ikke slettet.
- Nyt unikt simulationsklip `capture-frame-3`, automatisk tildelt Alpha.
- Eksisterende `capture-frame-2` blev requeued og derefter tildelt Alpha.
- `capture-video-frame-708` blev bevaret unresolved.

## Arbejdsgange og evidens

| Handling | Jeg forventede | Jeg så | Uden udviklerhjælp | Resultat | Evidens |
|---|---|---|---|---|---|
| Første overblik | Status, total, assigned, unresolved og seneste modtagelse inden 30 sek. | Toplinjen viste `Worker: Idle`; Overview viste statusfordelte tal, men intet samlet total- eller seneste-modtaget-tidspunkt. Standardvisningen var Videos. | Delvist | Observeret | [Første visning](../../temp/server-control-operator-20260911/01-first-look.png), [Overview](../../temp/server-control-operator-20260911/02-overview.png) |
| Se kliptal | Samlet, assigned og unresolved som entydige tal. | `Processed 2` og `Unresolved 1` var synlige; total skulle summeres, og `Processed` var ikke forklaret som assigned. | Delvist | Observeret | [Slut-overview](../../temp/server-control-operator-20260911/21-server-recovered.png) |
| Seneste klip | Et tydeligt tidspunkt for seneste modtagelse. | Kun workerens `Updated` og hvert klips `Captured`; intet `Received`/seneste klip. | Nej | Observeret | [Overview](../../temp/server-control-operator-20260911/02-overview.png) |
| Nyt klip til auto-assignment | Se klippet komme ind og ende hos korrekt rider. | Overview skiftede fra `Ready 1` til `Processed 1`; `capture-frame-3` endte hos Alpha via `tag-17`. Den hurtige processing gjorde `Processing` for kort til at blive fanget. | Ja | Observeret | [Ready](../../temp/server-control-operator-20260911/10-overview-change-9.png), [Processed](../../temp/server-control-operator-20260911/10-overview-change-10.png), [Detalje](../../temp/server-control-operator-20260911/12-processed-assignment-detail.png) |
| Forstå assignment | Bruger, hvorfor, datagrundlag og sikkerhed. | Bruger, tag, assignment-id, generisk reason og samlet score `0.72` var synlige. Motion/RSSI-værdier, tærskler og scorebidrag var ikke synlige. | Delvist | Observeret | [Assignment-detalje](../../temp/server-control-operator-20260911/12-processed-assignment-detail.png) |
| Forstå unresolved | Tydelig, bevaret status og forståelig årsag. | `Unknown rider`, `Unresolved` og årsagen om observation/motion/RSSI-gates var tydelige. Klippet kunne åbnes og var ikke fremstillet som tabt. | Ja | Observeret | [Unresolved-detalje](../../temp/server-control-operator-20260911/13-unresolved-detail.png) |
| Manuel omfordeling | Vælge en anden bruger fra klipdetaljen. | Ingen reassign/assign-handling i klipdetalje eller jobkø. | Nej | Observeret | [Processed detalje](../../temp/server-control-operator-20260911/12-processed-assignment-detail.png) |
| Ret brugeroplysninger | Redigere navn eller email. | Kun `Assign BearTag` og `Show videos`; ingen edit-handling. | Nej | Observeret | [Brugerhandlinger](../../temp/server-control-operator-20260911/15-user-detail-actions.png) |
| Skift BearTag-ejer i interval | Redigere/splitte eksisterende historik med konsekvensvisning. | Kun oprettelse af nye assignments. Et overlapforsøg blev blokeret med rå Pydantic-fejl og dokumentations-URL; eksisterende interval kunne ikke ændres. | Nej | Observeret | [Overlapforsøg](../../temp/server-control-operator-20260911/14-overlap-change-attempt.png) |
| Se berørte assignments | Liste over tidligere klip før ændringen gemmes. | Ingen impact preview eller liste over berørte klip. | Nej | Observeret | [Brugerhandlinger](../../temp/server-control-operator-20260911/15-user-detail-actions.png) |
| Genberegn og kontrollér | Genberegne de berørte assignments samlet og se før/efter. | Intet batch- eller processed-recompute-flow. Et unresolved job kunne requeues enkeltvis og blev derefter Processed hos Alpha. | Delvist | Observeret | [Før requeue](../../temp/server-control-operator-20260911/16-before-requeue.png), [Efter requeue](../../temp/server-control-operator-20260911/17-after-requeue.png), [Resultat](../../temp/server-control-operator-20260911/18-recomputed-assignment.png) |
| Serverfejl/stilstand | Tydelig årsag, påvirkning og næste handling. | `Failed to fetch` kom frem, men `Worker: Idle`, tid og tal stod stale. Ingen årsag, påvirkning eller recovery-instruks. Efter genstart kom UI'en tilbage. | Nej ved fejldiagnose | Observeret | [Før stop](../../temp/server-control-operator-20260911/19-before-server-stop.png), [Offline](../../temp/server-control-operator-20260911/20-server-offline.png), [Recovery](../../temp/server-control-operator-20260911/21-server-recovered.png) |
| Interne komponenter | Normal drift uden transport/storage-støj. | Ingen Box/storage/transport i primære flows. `Worker` og `Job queue` er tekniske begreber, mens manifestet er kollapset som standard. | Ja | Observeret | [Overview](../../temp/server-control-operator-20260911/02-overview.png), [Jobkø](../../temp/server-control-operator-20260911/11-job-queue-after-new-clip.png) |
| Kun sletning bekræftes | Requeue, brugerændring og assignment uden ekstra prompt. | User/tag/assignment blev gemt uden ekstra prompt, men Requeue krævede browserprompten `Requeue capture-frame-2?`. Ingen delete-handling fandtes at teste. | Nej | Observeret / deletion ikke testet | [Requeue før prompt](../../temp/server-control-operator-20260911/16-before-requeue.png) |

## Findings

### P0

Ingen observerede P0-fund. Klip blev bevaret gennem requeue og serverstop.

### P1

1. Manuel klipomfordeling mangler helt. En forkert assignment kan ikke rettes
   fra klipdetaljen.
2. Eksisterende brugeroplysninger og BearTag-intervaller kan ikke redigeres.
   Den centrale historikkorrektion er dermed blokeret.
3. Der er ingen konsekvensanalyse eller genberegning af tidligere processed
   assignments efter registryændringer. Operatøren kan ikke vide, hvilke gamle
   klip der fortsat er forkerte.

### P2

1. Overview mangler samlet kliptal, eksplicit assigned-tal og tidspunkt for
   senest modtagne klip.
2. Ved serverstop vises en generisk `Failed to fetch`, mens stale `Worker: Idle`
   og gamle tal forbliver synlige. Årsag, påvirkning og næste handling mangler.
3. Assignment-evidensen reduceres til en generisk tekst og én samlet score;
   datagrundlag og tærskler kan ikke efterprøves af operatøren.
4. Overlapfejl viser intern Pydantic-fejltekst og en ekstern dokumentations-URL
   i stedet for at forklare konflikten og foreslå en løsning.
5. Requeue kræver ekstra bekræftelse, selv om handlingen ikke sletter data.
6. Langsom submit/read-model-feedback gjorde det muligt at oprette en dublet,
   uden at UI'en viste en tydelig gemmer-status eller uniqueness-advarsel.

### P3

1. Standardvisningen er Videos frem for Overview. Status er stadig synlig i
   toplinjen, men operatøren skal tage et ekstra klik for fuldt driftsblik.
2. `Worker` og `Job queue` er mildt implementeringsorienteret sprog, selv om
   transport og storage ellers er holdt ude af normaloplevelsen.

## Bestået, fejlet og ikke testet

Bestået:

- lokal deployment, build og automatiske tests;
- normal driftsstatus inden 30 sekunder;
- nyt klip fra Ready til automatisk assignment;
- synligt unresolved-klip og bevaret medie;
- individuel requeue med synligt nyt resultat;
- recovery efter lokalt serverstop;
- transport/storage dominerer ikke de primære views.

Fejlet eller kun delvist:

- total/assigned/latest-received-overblik;
- forklaring af assignmentens konkrete datagrundlag;
- manuel klipomfordeling;
- brugerredigering;
- ændring af eksisterende BearTag-ejerskab;
- impact preview og genberegning af tidligere processed assignments;
- handlingsanvisende serverfejl;
- kravet om kun ekstra bekræftelse ved sletning.

Ikke testet:

- faktisk sletning og dens bekræftelse, fordi ingen delete-handling var synlig;
- Box/produktion og rigtige credentials, bevidst uden for scope;
- multi-page skala, samtidige operatører og mobile flows;
- fysisk BearTag-hardware og reelle klipmodtagelser;
- `Processing`-tilstanden som langvarig UI-tilstand, fordi lokal behandling var
  hurtigere end refresh-intervallet.

## Sluttilstand

- Service kører igen med samme commit og `config/server.local.yaml`.
- Worker: `Idle`, `processRunning: true`.
- Kø: Ready 0, Processing 0, Processed 2, Unresolved 1, Failed 0.
- Ingen klip eller users blev slettet.
- Tre lokale testusers, to tags og to aktive assignments er bevaret.
- Det ekstra Alpha-fixture er bevidst bevaret i overensstemmelse med forbuddet
  mod sletning af brugerdata.
- Eksisterende `.tmp/` er urørt. Testharnesses, screenshots og captures ligger i
  `temp/server-control-operator-20260911/`.

## Tre vigtigste næste handlinger

1. Lever ét sikkert korrektion-flow: redigér/split BearTag-historik, vis berørte
   assignments før gemning, og genberegn dem med tydeligt før/efter-resultat.
2. Tilføj manuel klipomfordeling og redigering af brugerdata direkte i UI'en,
   uden ekstra bekræftelse for ikke-slettende handlinger.
3. Gør Overview operationelt: samlet/assigned/unresolved, seneste modtagelse og
   en ærlig offline-tilstand med årsag, påvirkning og næste handling.
