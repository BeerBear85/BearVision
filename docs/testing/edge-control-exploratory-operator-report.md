# Edge Control – exploratory usability-test som kabelparkoperatør

## Resumé

Den nyeste version er væsentligt lettere at forstå end den tidligere testede version. Ved første blik fortæller portalen nu klart, at simulationen er klar, at fysisk udstyr ikke kontrolleres, og at ingen handling kræves under normal drift. Start, kameraaktivitet, reload under drift og normalt stop fungerede gennem brugerfladen.

Samlet usability-score: **7/10**. Normalforløbet er brugbart uden ingeniørviden. De største resterende problemer findes ved fejl og recovery: En gennemført kørsel kunne samtidig vise en rød fejl om et manglende capture, og handlingerne efter et forbindelsestab kunne pege i modstridende retninger. Ingen P0 blev observeret. Tre P1- og fire P2-findings er beskrevet nedenfor.

Dette er en simuleret personas vurdering, ikke et studie med faktiske kabelparkoperatører.

## Testmiljø, tidspunkt og begrænsninger

- Testet 9. september 2026 cirka 14:12–14:32, Europe/Copenhagen.
- Repository: `D:/Repositories/BearVision_2`, commit `4a4462f18f28a7106174d9544b9871855678a27f`. `git fetch origin` bekræftede, at `HEAD` og `origin/master` var identiske.
- Lokal redeploy efter `apps/edge-control/README.md`: `corepack pnpm install --frozen-lockfile`, `corepack pnpm build` og `node server/server.mjs`. Portalen blev åbnet på `http://localhost:4310` med serveren bundet til `127.0.0.1`.
- Testdata, kø, logs og tilstand blev isoleret under `temp/operator-exploration-20260909`.
- Browseren blev betjent gennem den synlige UI med labels, knapper og scenarievælger. Viewports: normal desktop og 320 × 740 CSS-pixels.
- Kode og kommandoværktøjer blev kun brugt til at hente, bygge, starte, afbryde og kontrollere miljøet. De tæller ikke som bevis for UI-forståelighed.
- Som teknisk miljøkontrol bestod 58/58 servertests og 9/9 browsertests. Det ændrer ikke de observerede usability-findings.
- Ingen hardwaretilstand blev valgt, intet fysisk udstyr blev startet, og der blev ikke foretaget ekstern deployment eller implementeret rettelser.

“Observeret” betyder, at resultatet var synligt i portalen efter en faktisk UI-handling. “Udledt” er konsekvensen vurderet fra operatørpersonaen. “Ikke testet” bruges, når miljøet ikke gav pålideligt UI-bevis.

## Gennemførte brugerforløb

| Valg og forventning før handling | Faktisk synligt resultat | Status og konsekvens |
| --- | --- | --- |
| Åbn portalen; forvent hurtigt svar på klar, kører, problem og næste handling. | “Simulation is ready to run”, Camera Idle, Background 0 active/queued, Connection Live og “Physical equipment is not checked in simulation mode.” | Observeret. Første blik er entydigt, og operatøren kan starte uden at gætte på readiness. |
| Vælg simulation og et egnet normalforløb; forvent et forståeligt scenarie. | Simulation var valgt. “single rider success” var let at afkode; de øvrige navne indeholdt bl.a. yolo, 60fps og regression. | Observeret. Et enkelt normalscenarie er let; resten kræver teknisk gætteri. |
| Dobbeltklik Run scenario; forvent højst én start. | Knappen skiftede til deaktiveret “Starting…”. Kun én kørsel blev synlig. | Observeret i dette forsøg. Ingen synlig dobbeltstart. |
| Følg normal drift; forvent at forstå optagelse og kliparbejde. | “BearVision is recording riders”, Camera Capturing og forklaringen om, at nye klip optages, mens tidligere klip kan færdiggøres i baggrunden. | Observeret. Forholdet mellem kamera og baggrundsarbejde er klart. |
| Stop mens kameraet er aktivt; forvent sikker kvittering og forståelse af klippets skæbne. | Stopknappen blev straks deaktiveret. Inden cirka 0,8 sekund stod portalen igen som Simulation ready; historikken viste Stopped og 0 outputs. | Observeret. Stoppet lykkedes, men UI forklarede ikke, om det aktive klip blev kasseret eller aldrig nåede at blive oprettet. |
| Skift scenarie efter et resultat; forvent, at gammelt preview fjernes. | Tidligere klip og overlay blev fjernet, og det nye scenaries preview blev vist. | Observeret. Det tidligere fund om gammelt preview er rettet. |
| Reload under aktiv testmovie1-kørsel; forvent at kunne fortsætte aflæsningen. | Efter en kort Loading/Reconnecting-tilstand kom samme låste scenarie tilbage som Running med Live-forbindelse. | Observeret. Aktiv tilstand blev genskabt korrekt. |
| Lad den genindlæste videokørsel afslutte; forvent ét sammenhængende resultat. | Recent runs viste Completed, 4 outputs og 0 failures, samtidig med et rødt “Action failed – Capture does not exist.” Tekniske detaljer viste “Action: load tracking data” og `MEDIA_NOT_FOUND`. | Observeret. Operatøren kan ikke sikkert afgøre, om resultatet er gyldigt. |
| Kopiér supportdetaljer; forvent en enkel delingshandling. | Knappen skiftede synligt til “Copied”. Selve clipboard-indholdet blev ikke verificeret. | Observeret UI-kvittering; faktisk deling er ikke testet. |
| Brug portalen ved 320 × 740; forvent at se kamera, kø, readiness og handlinger uden sidelæns søgning. | Kamera, Background clips, Readiness og pipeline-detaljer blev vist som mobile kort. Under aktiv drift var status og stop synlige. En lille vandret scrollbar og afkortet langt filnavn blev også set. | Observeret. Den gamle skjulte pipeline er rettet; der er mindre mobilfriktion tilbage. |
| Fremprovokér en hel runtime-kamerafejl; forvent forklaring af problem, påvirkning og handling. | “The simulated camera could not record”, “No new test clips are being captured” og “Restart the runtime to continue the test.” Fejlen overlevede reload. | Observeret. Fejlteksten er operatørvenlig og tekniske detaljer er sekundære. |
| Kom videre efter kamerafejlen; forvent at kunne afslutte testen eller vælge et andet scenarie. | Scenariet var låst, og eneste primære handling var Restart runtime. Restart startede samme fejlscenarie igen; efter Stop blev opsætningen frigivet. | Observeret. Recovery virker, men kræver en ulogisk omvej. |
| Afbryd kontrolserveren under start og tryk Stop; forvent at få at vide, om stop er modtaget. | “We could not confirm that BearVision stopped”, mulig fortsat optagelse, seneste kendte tilstand, Copy support details og Try stop again. | Observeret. Den tidligere “Failed to fetch”-fejl er rettet. |
| Genskab serveren og vælg Try stop again; forvent, at UI afklarer den nu kendte tilstand. | Portalen kendte nu processen som tabt og viste en vedvarende fejl med Restart runtime, men Try stop again gav “runtime process is not active” og generisk “Try again”. Restart runtime virkede; efterfølgende Stop gav Simulation ready. | Observeret. Recovery lykkes, men de to handlinger giver modstridende vejledning. |
| Start Blender-scenarie; forvent forklaring ved længere forberedelse. | Efter 21 sekunder stod “BearVision is starting” og “Preparing the selected runtime. No action is needed yet.” Kørsel var færdig inden næste aflæsning cirka 25 sekunder senere med 1 output og 0 failures. | Observeret. Beskeden reducerer usikkerhed, men der mangler forventet varighed eller seneste fremdrift. |

## Det fungerede godt

- Første blik besvarer nu de fire centrale operatørspørgsmål med én samlet status og almindeligt sprog.
- Simulation-readiness er korrekt beskrevet som “Not used” frem for “Not checked”.
- Startknappen kvitterer straks og låser gentagne klik; scenarievalg er låst under drift.
- “Recording riders”, Camera Capturing og forklaringen om baggrundsklip gør normal drift forståelig.
- Reload genskaber både aktiv kørsel og vedvarende fejl.
- Scenarieskift nulstiller tidligere preview og overlay.
- Kamerafejlen forklarer både problem, påvirkning og næste handling, mens tekniske detaljer kan åbnes separat.
- Et ubekræftet stop er nu knyttet til den konkrete handling og forklarer, at BearVision muligvis stadig optager.
- Mobilvisningen gør kamera, kø og readiness synlige uden vandret rulning inde i pipelinekortene.

Sammenlignet med testen 8. september er de tidligere findings om modstridende readiness, gammelt preview, rå “Failed to fetch”, teknisk kamerafejl, skjult mobilpipeline og manglende supportkopiering tydeligt forbedret eller rettet.

## Prioriterede findings

### F01 · P1 · En gennemført kørsel vises samtidig som fejlet handling

- Situation og reproduktion: Vælg `wakeboard-testmovie1-yolo`, start, genindlæs siden under kørsel, og lad forløbet afslutte.
- Konkret UI-evidens: Recent runs viste “Completed”, “4 outputs · 0 failures” og et konkret extracted clip. Samtidig stod et rødt banner med “Action failed – Capture does not exist.” Under Technical details stod “Action: load tracking data” og `MEDIA_NOT_FOUND`.
- Jeg forventede en entydig afslutning, men jeg så både et vellykket resultat og en fejl, der kunne forstås som fejl i hele kørslen.
- Konsekvens: Operatøren kan kassere et brugbart resultat eller overse, at en del af resultatvisningen faktisk mangler. Om selve outputtet var korrekt, blev ikke verificeret.
- Forslag: Afgræns fejlen til den berørte visning: “Kørslen er gennemført, men tracking-visningen kunne ikke indlæses.” Bevar Completed som hovedstatus, angiv hvilke outputs der stadig kan bruges, og tilbyd “Prøv at indlæse visning igen”.
- Evidens: [afslutning med fejl](evidence/edge-control-operator-20260909/11-long-running.png), [tekniske detaljer](evidence/edge-control-operator-20260909/12-support-details.png).

### F02 · P1 · Recovery efter forbindelsestab giver modstridende handlinger

- Situation og reproduktion: Start et scenarie, afbryd kontrolserveren, tryk Stop, genskab serveren, og tryk Try stop again.
- Konkret UI-evidens: Første besked forklarede korrekt, at stop ikke var bekræftet. Efter reconnect viste portalen en persistent fejl med “Restart runtime”, mens den gamle “Try stop again” stadig var aktiv. Klik på den gav “runtime process is not active” og samtidig rådet “Try again”.
- Jeg forventede, at portalen ville afklare den gamle stopkommando ud fra den nye servertilstand, men jeg så to handlinger, der pegede i hver sin retning.
- Konsekvens: Operatøren kan blive ved med at gentage et stop, som ikke længere giver mening, eller være i tvivl om en genstart er sikker. Restart virkede i testen, så en central opgave var ikke blokeret.
- Forslag: Når reconnect fastslår, at processen er væk, skal det ubekræftede stop afsluttes automatisk som “Processen er stoppet/forbindelsen gik tabt”. Fjern Try stop again og vis én recovery-handling med forklaring af, hvad genstart gør.
- Evidens: [ubekræftet stop](evidence/edge-control-operator-20260909/22-unconfirmed-stop-later.png), [fejl efter nyt stopforsøg](evidence/edge-control-operator-20260909/23-retry-stop-after-reconnect.png), [recovery afsluttet](evidence/edge-control-operator-20260909/24-recovery-after-restart.png).

### F03 · P1 · En fejlet test kan kun forlades ved at genstarte samme scenarie

- Situation og reproduktion: Kør kamerafejlscenariet til Failed, og forsøg derefter at vælge et andet scenarie.
- Konkret UI-evidens: Runtime var fejlet og kameraet Idle, men mode og scenarievælger var låst. Kun Restart runtime var tilgængelig. Den startede samme scenarie igen; et efterfølgende Stop frigav valget.
- Jeg forventede at kunne afslutte den fejlede test og vælge en anden prøve, men jeg så kun en handling, der gentog den fejlende prøve.
- Konsekvens: En operatør kan havne i en gentagelsesløkke eller tro, at genstart er nødvendig, selv når målet er at afbryde testen. Fejlhistorikken blev bevaret, så der blev ikke observeret datatab.
- Forslag: Tilføj “Afslut fejlet kørsel” eller “Vælg andet scenarie”. Forklar, at fejl- og supportoplysninger bevares.
- Evidens: [kamerafejl](evidence/edge-control-operator-20260909/18-camera-failure.png), [fejl efter reload](evidence/edge-control-operator-20260909/19-failure-restored.png), [opsætning frigivet efter omvej](evidence/edge-control-operator-20260909/20-recovery-idle.png).

### F04 · P2 · Scenarielisten bruger tekniske navne

- Situation og reproduktion: Åbn Scenario-listen og sammenlign valgmulighederne.
- Konkret UI-evidens: Navne som “wakeboard testmovie1 yolo” og “wakeboard fs360 60fps blender regression · Blender”; preview viser bl.a. `bear_tag`, Kalman + RTS og Butterworth.
- Jeg forventede at vælge efter formål, men jeg så implementerings- og testnavne.
- Konsekvens: Operatøren kan vælge en langsom eller forkert prøve. “single rider success” var undtagelsen, der var let at forstå.
- Forslag: Vis operatørnavn, formål, forventet varighed og forventet resultat. Flyt filnavn og algoritmer til Technical details.
- Evidens: [første skærm](evidence/edge-control-operator-20260909/01-first-look.png), [Blender under opstart](evidence/edge-control-operator-20260909/25-blender-starting-21s.png).

### F05 · P2 · Stop forklarer ikke det aktive klips skæbne

- Situation og reproduktion: Stop et normalforløb, mens Camera viser Capturing.
- Konkret UI-evidens: Stopknappen blev deaktiveret, og portalen gik hurtigt til Simulation ready. Recent runs viste Stopped og 0 outputs uden forklaring.
- Jeg forventede at få at vide, om det aktive klip blev færdiggjort eller kasseret, men jeg så kun nul output.
- Konsekvens: Operatøren kan ikke planlægge et stop med sikker viden om det igangværende klip. Faktisk kliptab er ikke bevist.
- Forslag: Vis under stop og i historikken: “Aktiv optagelse blev kasseret” eller “Aktivt klip færdiggøres i baggrunden”, baseret på den faktiske politik.
- Evidens: [stop under kameraaktivitet](evidence/edge-control-operator-20260909/05-stop-after-camera-activity.png), [stopresultat](evidence/edge-control-operator-20260909/06-stop-result.png).

### F06 · P2 · 320-pixelsvisningen har en lille vandret overflow

- Situation og reproduktion: Brug 320 × 740 og start et videoscenarie.
- Konkret UI-evidens: De centrale statuskort var synlige, men nederst i viewporten sås en vandret scrollbar; et langt fil-/scenarienavn blev afkortet.
- Jeg forventede en helt lodret mobilvisning, men jeg så mindre sidelæns overflow.
- Konsekvens: Den centrale opgave kan gennemføres, men siden virker mindre stabil og lange resultater er svære at identificere.
- Forslag: Undgå bredder baseret på hele viewporten, når den lodrette scrollbar optager plads, og tillad kontrolleret linjeskift i lange navne.
- Evidens: [mobil under drift](evidence/edge-control-operator-20260909/15-mobile-active.png), [mobil stopresultat](evidence/edge-control-operator-20260909/17-mobile-stop-result.png).

### F07 · P2 · Længere opstart mangler forventet varighed

- Situation og reproduktion: Start `wakeboard-fs360-60fps-blender-regression`.
- Konkret UI-evidens: Efter 21 sekunder stod “Preparing the selected runtime. No action is needed yet.” med en løbende tid. Kørsel gennemførte inden cirka 47 sekunder.
- Jeg forventede en grov forventet ventetid eller tegn på fremdrift, men jeg så kun, at jeg fortsat skulle vente.
- Konsekvens: Operatøren kan afbryde en sund, langsom start. Den nye tekst er dog klart bedre end den tidligere rene status “Initializing”.
- Forslag: Vis forventet interval for det valgte scenarie og tidspunkt for seneste fremdrift; efter en overskridelse skal UI forklare næste handling uden opdigtede procenttal.
- Evidens: [21 sekunders opstart](evidence/edge-control-operator-20260909/25-blender-starting-21s.png), [gennemført Blender-kørsel](evidence/edge-control-operator-20260909/26-blender-completed.png).

## Fejlmeddelelser og næste handling

| Fejl | Hvad er galt? | Hvad er påvirket? | Kan driften fortsætte? | Konkret handling | Supportdetaljer |
| --- | --- | --- | --- | --- | --- |
| Simuleret kamerafejl | Ja, kameraet kunne ikke optage. | Ja, ingen nye testklip optages. | Ja, efter runtime restart. | Ja, Restart runtime. | Ja, sammenklappet. |
| Stop uden kontrolforbindelse | Ja, stop kunne ikke bekræftes. | Ja, BearVision kan stadig optage. | Usikkert med vilje; seneste tilstand vises. | Ja, kontrollér forbindelse og prøv igen. | Ja, kopi-knap. |
| Stopforsøg efter reconnect | Delvist; processen er ikke aktiv. | Nej, forholdet til den tidligere stopkommando forklares ikke. | Nej, Restart og Try again konkurrerer. | Uklart. | Ja, men operatørteksten er generisk. |
| Manglende capture ved Completed | Delvist; capture findes ikke. | Kun i tekniske detaljer: tracking data. | Uklart, fordi Completed og rød fejl står samtidig. | Generisk Try again/contact support. | Ja, `MEDIA_NOT_FOUND` og handling kan kopieres. |

En fejl i ét enkelt baggrundsklip blev ikke fremprovokeret pålideligt og er derfor ikke vurderet. Hele runtime-fejlen blev observeret gennem kamerafejlscenariet.

## Scorer

| Område | Score | Begrundelse |
| --- | --- | --- |
| Overblik | 8/10 | Første blik er klart og handlingsorienteret; fejl efter afslutning kan stadig modsige hovedresultatet. |
| Start og readiness | 8/10 | Simulation-readiness og start er tydelige. Scenarienavne og længere opstart kræver fortolkning. |
| Forståelse under drift | 8/10 | Kamera og baggrundsarbejde forklares godt, og reload bevarer drift. Resultatfejlen svækker tilliden. |
| Stop og recovery | 6/10 | Normalt stop virker, og ubekræftet stop forklares. Recovery-handlinger efter reconnect og terminal fejl er uklare. |
| Fejlmeddelelser og næste handling | 7/10 | Kamera- og første stopfejl er gode; de efterfølgende recovery- og mediafejl er ikke entydige. |
| Samlet | 7/10 | Normal drift er brugbar, men de sjældne situationer, hvor operatøren mest behøver sikker vejledning, har fortsat P1-problemer. |

## De tre vigtigste anbefalede ændringer

1. Gør afslutningsresultatet entydigt: adskil en vellykket kørsel fra fejl i en bestemt resultatvisning, og sig præcist hvilke outputs der kan bruges.
2. Saml recovery efter fejl og reconnect i én autoritativ handling. Fjern forældede stopforsøg, og tilbyd at afslutte en fejlet kørsel uden at genstarte samme scenarie.
3. Gør scenarievalg og ventetid operatørvenlige med formål, forventet varighed og forventet resultat; behold fil- og algoritmenavne i tekniske detaljer.

## Ikke testet eller ikke bekræftet

- Hardwaretilstand, fysisk kamera, BLE/BearTag, hardware-readiness, blokeret hardwarestart og advarselskvitteringer.
- En isoleret fejl i ét klip, retry af et klip og om resten af driften kan fortsætte i netop det tilfælde.
- Force stop efter udløbet graceful-stop-timeout.
- Stop med et observeret aktivt Processing-, Packaging- eller Uploading-job.
- Faktisk kvalitet og afspilning af alle genererede videoer og trackingdata. UI viste fil og overlay, men mediefilernes indhold blev ikke valideret.
- Indholdet i clipboard efter Copy support details og faktisk deling med support.
- Ægte mobiltelefon/touch, andre browsere, skærmlæser, længere tids drift og flere samtidige operatører.
- Fysisk databevarelse eller adfærd efter strømudfald.

## Testmiljøets sluttilstand

Sidste UI-observation viste Simulation ready, single rider success valgt, Camera Idle, Background 0 active/queued, Connection Live og aktiv Run scenario. Det midlertidige kamerafejlscenarie var fjernet fra scenarielisten.

Browserfanen og alle lokale processer startet til testen blev stoppet. Kontrol kl. 14:39:55 fandt ingen resterende egne processer og ingen lytter på port 4310. Den midlertidige scenariefil blev fjernet. Build-output, isoleret testtilstand, logs og evidens er bevaret; applikationskoden er ikke ændret.

- [Sidste UI-skærm](evidence/edge-control-operator-20260909/27-final-idle.png).
- [Sidste UI-tekst](evidence/edge-control-operator-20260909/27-final-idle.txt).
- [Verificeret oprydning](evidence/edge-control-operator-20260909/cleanup.json).
