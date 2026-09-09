# Edge Control – exploratory usability-test som kabelparkoperatør

## Resumé

Jeg kan finde simulation, starte et forløb, se kameraaktivitet og stoppe via portalen. Jeg kan også genfinde en aktiv kørsel og en vedvarende fejl efter genindlæsning. Men jeg skal sammenholde flere statusfelter og forstå tekniske ord for at afgøre, om Bear Vision er klar, arbejder eller kræver hjælp.

Samlet usability-score: **5/10**. Betjeningen fungerer i de afprøvede start/stop-forløb, men status, gamle preview-oplysninger og fejlvejledning skaber væsentlig tvivl. Ingen P0 er bekræftet i det afprøvede scope. Syv P1- og tre P2-findings er beskrevet nedenfor. Det er en simuleret personas vurdering, ikke et studie med faktiske kabelparkoperatører.

Det centrale spørgsmål fra operatøren er: Når skærmen siger “Review the technical details”, hvad skal jeg konkret kontrollere, hvis jeg ikke forstår teknikken?

## Testmiljø og metode

- Dato: 8. september 2026. Miljøopsætning fra cirka 21:41; UI-observationer 21:44:08–21:54:53; oprydning verificeret 21:54:58, Europe/Copenhagen, UTC+02:00.
- Repository: `D:/Repositories/BearVision_2`, commit `e76f67104f293a56f697a6884cfe273c0ca1a201`.
- Portal: `http://localhost:4310`, server bundet til `127.0.0.1`.
- Windows, Node 22.18.0, pnpm 10.34.5, projektets Python 3.12.4 i `.venv`, Chrome 152.0.7977.77.
- Browser: rigtig Chrome i headless-tilstand, betjent gennem Playwright med synlige labels, knapper, scenarievælger og navigation. UI-evidens er tilgængelighedstræer og skærmbilleder; udvalgte skærmbilleder blev også visuelt inspiceret.
- Viewports: 1366 × 900 og 320 × 740 CSS-pixels. Fuldsidebilleder viser mere end det, der kan ses uden lodret rulning. På desktop er billedbredden i flere PNG-filer 1351 pixels på grund af scrollbar.
- Startvejledning: `apps/edge-control/README.md`. `corepack pnpm install --frozen-lockfile` og `corepack pnpm build` lykkedes. Serveren blev startet med `node server/server.mjs`, som er indholdet af README'ens `pnpm serve`-script.
- Kontroltilstand, lokal kø, serverlogs og browserprofil blev isoleret under `temp/operator-exploration-20260908`. `BEARVISION_CONTROL_STATE_PATH`, `BEARVISION_LOCAL_QUEUE_ROOT`, `BEARVISION_CAPTURE_ROOT` og `BEARVISION_SCRATCH_ROOT` blev sat. Simulationens syntetiske capture-adapter vælger dog egne mapper under `temp/captures/.simulation-*`; capture-miljøvariablen isolerer derfor ikke alle simulationsartefakter. Dette blev afklaret ved teknisk efterkontrol.
- Ingen hardwaretilstand blev valgt, intet fysisk udstyr startet, intet deployet og ingen applikationsrettelser implementeret.

Den almindelige shell, Node REPL og CUA-browserintegration fejlede ved sandbox-opstart. Miljøopsætning og Playwright blev derfor kørt gennem det tilgængelige kommandoværktøj med godkendt adgang uden sandbox. Dette er en værktøjsbegrænsning, ikke et UI-fund.

Missionen var: “Start, forstå driften og stop uden teknisk hjælp”. Før væsentlige handlinger blev forventningen noteret. Det korte første forløb afsluttede mellem observationer; det førte til en gentagelse med tættere observation. Gamle preview-oplysninger førte til afprøvning af scenarieskift og genindlæsning. Lang opstart førte til stop- og forbindelsesfejltest. Fejlene førte til afprøvning af Diagnostics og recovery.

### Evidens og afgrænsning

“Observeret” betyder synligt i portalen ved en faktisk UI-handling eller efterfølgende aflæsning. “Udledt” er min vurdering af betydningen for operatøren. “Ikke bekræftet” betyder, at forsøget ikke gav det ønskede testresultat. Skærmbilleder beviser ikke alene klikforløb, procesadfærd eller fysisk optagelse.

Kildekode blev kun læst til opstart, sikker afgrænsning og kontrol af simulationsmuligheder. Den tæller ikke som bevis for UI-forståelighed. Personaens tekniske forståelse blev ikke løftet af denne viden.

Der blev brugt tre eksisterende scenarier: `single-rider-success.yaml`, `wakeboard-testmovie1-yolo.yaml` og `wakeboard-fs360-60fps-blender-regression.yaml`.

To midlertidige scenarier blev kopieret fra single-rider-scenariet med henholdsvis `storage_upload: true` og `camera_capture: true`, uden forventningsblok og med en ekstra tag-observation ved T+30. De findes som reproduktionsbilag i evidensmappen, men er fjernet fra scenariekataloget:

- [Uploadfejl-fixture](evidence/edge-control-operator-20260908/zz-operator-upload-failure.yaml).
- [Kamerafejl-fixture](evidence/edge-control-operator-20260908/zz-operator-camera-failure.yaml).

Mine første fixtures havde fejlagtigt et `title`-felt, som Python afviste. To kørsler gav derfor valideringsfejl. Det er en fejl i testopsætningen, ikke bevis for uploadfejl eller fejl i et medfølgende scenarie. Feltet blev fjernet før de gyldige fejltests. UI'ens generiske fejlvisning blev observeret under disse forsøg, og den tilsvarende vejledning blev efterfølgende genfundet ved den gyldige kamerafejl.

## Første blik

| Operatørspørgsmål | Hvad jeg så | Vurdering |
| --- | --- | --- |
| Er Bear Vision klar? | Grøn Readiness-markør, men “Readiness: Not checked”; ingen særskilt readiness-knap i simulation. | Udledt: uklart, om jeg mangler at gøre noget. |
| Kører systemet? | “Idle”, “Runtime process: Idle”, “Camera: Idle” og aktiv “Run scenario”. | Udledt: tilstrækkeligt tegn på, at der ikke er en aktiv kørsel. |
| Er der et problem? | Ingen fejlmeddelelse, nul fejlede klip, “Control connection: Live”. | Observeret: ingen synlig fejl; det beviser ikke readiness. |
| Forventes en handling? | Scenarie er valgt, og “Run scenario” er den fremtrædende handling. | Udledt: jeg skal starte, men readiness-feltet gør mig usikker. |

Evidens: [indlæst første skærm](evidence/edge-control-operator-20260908/02-idle-loaded.png). Den kortvarige “Loading/Reconnecting”-skærm ved første indlæsning blev ikke klassificeret som en vedvarende fejl.

## Gennemførte brugerforløb og beslutninger

| Valg og forventning før handlingen | Faktisk synligt resultat | Status og konsekvens |
| --- | --- | --- |
| Vælg Simulation og single rider success; forvent et enkelt normalt forløb. | Simulation var allerede valgt. Scenariet kunne vælges, og Run scenario var aktiv. | Observeret. Starten er let at finde. |
| Kontrollér readiness før start; forvent kontrol eller forklaring. | System viste Not checked. Ingen readiness-kontrol var synlig i simulation. | Observeret søgning; readiness kan ikke kaldes kontrolleret. Uklar næste handling. |
| Tryk Run scenario; forvent kvittering og fremdrift. | Starting… blev deaktiveret, derefter Initializing. Efter cirka syv sekunder stod første kørsel stadig Initializing. | Observeret. Trykket kvitteres, men ventetidens årsag forklares ikke. |
| Vend tilbage efter cirka 15 sekunders observationspause; forvent et tydeligt resultat. | Idle i toppen; Completed og “1 outputs · 0 failures” under Recent runs. | Observeret. Afslutning findes, men ikke i hovedstatus. |
| Genindlæs efter denne afslutning; forvent historik bevaret. | Completed blev bevaret; Diagnostics gik fra 25/25 til 0/0; preview skiftede til Scenario source. | Observeret efter afsluttet kørsel. Dette var ikke en genindlæsning under drift. |
| Dobbeltklik Run scenario; forvent højst én ny kørsel. | Starting… deaktiveret. Én ny Completed-post kom til. Monitoring, Running og Camera: Capturing blev set. | Observeret i denne afprøvning. Ingen synlig dobbeltkørsel; ikke en generel garanti mod alle gentagne klik. |
| Følg kamera og klip; forvent forståelig forbindelse mellem optagelse og resultat. | Camera: Capturing, capture-frame-2 og senere Extracted clip. Efter afslutning viste køen igen 0 completed, mens historikken viste 1 output. | Observeret. Optagelsesstatus forstås bedre end resultatets betydning. Processing/Packaging/Uploading blev ikke fanget som vedvarende aktive trin. |
| Vælg testmovie1; forvent preview for det nye scenarie. | Gamle capture-frame-2.mp4, 0 KiB, T+9.0 og overlay blev stående, mens komponentlabels skiftede. | Observeret. Risiko for at forveksle gammelt resultat med nyt scenarie. |
| Tryk Scenario source og genindlæs under aktiv testmovie1-kørsel; forvent aktiv status bevaret. | Før reload: Initializing 15s. Efter reload: samme valgte, låste scenarie, Monitoring og Runtime process Running. Gamle klipfelter var væk. | Observeret under aktiv kørsel. Årsagen til faseskiftet kan ikke tilskrives reload. |
| Dobbeltklik Stop runtime på testmovie1; forvent stopkvittering. | Stopping, stopknappen forsvandt, senere Idle og en Stopped-post med 0 outputs. Kameraet var Idle ved stoptrykket. | Observeret. Dette beviser stop af aktiv simulation, ikke stop under klipbehandling. |
| Skift til 320 pixels; start single rider og stop ved Camera: Capturing. | Scenarievalg og stop virkede. Stopping blev fulgt af Idle og Stopped med 0 outputs efter cirka 1,5 sekund. Kameratrinnet var delvist skjult i kortet. | Observeret. Handlingen kan gennemføres, men klippets skæbne og kameraoverblik er uklare. |
| Åbn Diagnostics efter stop; forvent supportinformation. | Rå hændelser og logniveau-filter; tidspunkter som LIVE og T+4.0. Ingen synlig kopi-/eksportknap. | Observeret. Support kan få et manuelt skærmbillede, men operatøren får ikke en færdig hændelsesrapport. |
| Start Blender-scenariet; forvent forklaring, hvis opstart varer længe. | Initializing blev set ved 22s, 30s, 51s og 1m 8s. Ingen forklaring eller forventet ventetid. | Observeret. Om beregningen ville lykkes ved længere ventetid, blev ikke afprøvet. |
| Simulér fem sekunders offline i browseren; forvent tydelig forbindelsesstatus. | Control connection blev ved med at vise Live. | Observeret UI; forsøget bekræfter ikke, at en eksisterende eventforbindelse blev afbrudt. |
| Klik Stop runtime mens browseren er offline; forvent forklaring på ikke-modtaget stop. | Rød “Failed to fetch”; Stop runtime var fortsat mulig, og Control connection viste Live. | Observeret kommando-fejl. Omfang og næste handling mangler. |
| Genskab forbindelsen og tryk Stop igen; forvent recovery. | Fejlbanneret forsvandt ved det nye forsøg; senest seks sekunder senere viste portalen Idle og Stopped. | Observeret. Stop virkede efter nyt tryk; automatisk gensendelse er ikke bekræftet. |
| Start uploadfejl-fixture; forvent ét fejlet klip. | Først ugyldigt title-felt. Efter rettelse: Clip uploaded, 1 completed og senere Completed med 0 failures. | Uploadfejl ikke bekræftet. Intet grundlag for at vurdere retry af enkeltklip. |
| Start gyldig kamerafejl-fixture; forvent tydelig kamerafejl. | Først Stopped samtidig med Runtime process Running, uden fejlkort. Senere Failed, Exited og Persistent failures med “injected camera capture failure”. | Observeret. Forsinket fejlvisning og modstridende status i dette replay-forløb. |
| Åbn Technical details, vælg Error-filter og genindlæs; forvent brugbar, bevaret supportevidens. | Fejl-ID, Operation: Not available, fejltekst og Attempts: 1. Error-filter fjernede stacktrace, men beholdt flere almindelige hændelser. Efter reload blev fejlkortet bevaret, Diagnostics blev 0/0. | Observeret. Fejlens eksistens bevares bedre end dens hændelseshistorik. |
| Kom videre fra en afsluttet fejl; forvent mulighed for andet scenarie. | Kun Restart runtime var tilgængelig; scenarievælgeren var låst. Genstart fulgt straks af Stop gav Idle, hvorefter single rider kunne vælges. | Observeret omvej. Ingen P0-blokering bekræftet, men den nødvendige omvej fremgår ikke. |

## Det fungerede godt

- Simulation er tydeligt valgt og gentages i preview. Der var ingen nødvendig overgang til hardware for at starte de normale testforløb.
- Den primære knap er let at finde. Starting… og Stopping giver synlig kvittering; ændring af scenarie er låst under en aktiv kørsel.
- Live og Background queue er særskilte områder. Camera: Capturing gav konkret information om optagelse; farver står sammen med tekst.
- Aktivt scenarie og driftstilstand blev genskabt efter reload. En vedvarende kamerafejl forsvandt heller ikke ved reload.
- Fejlkort har tydelig rød markering, komponent og tidspunkt. Technical details er sammenklappet som udgangspunkt.
- Start og stop kunne gennemføres ved 320 pixels; hele siden havde ingen vandret overflow. Problemet er de enkelte kort og navigationen.

## Prioriterede findings

Prioriteter: P0 = central opgave umulig eller alvorlig forkert handling; P1 = sandsynlig misforståelse af tilstand/næste handling; P2 = mindre uklarhed eller friktion. UI-fakta nedenfor er observerede. De forventede konsekvenser for faktiske operatører er udledte persona-hypoteser.

### F01 · P1 · Statusfelterne giver ikke ét entydigt svar

- Situation/reproduktion: Åbn frisk simulation. Sammenhold Readiness i Live-kortet med System. Gentag med den vedlagte kamerafejl-fixture og se status før den endelige fejl.
- UI-evidens: Grøn Readiness, men Not checked; i fejlreplayet Stopped og Runtime process Running samtidig med aktiv Stop runtime-knap. Ingen fejloplysning i hovedfladen på dette tidspunkt.
- Jeg forventede ét svar på, om Bear Vision var klar eller stoppet, men jeg så felter, der pegede i forskellige retninger.
- Konsekvens: Jeg kan enten springe en nødvendig kontrol over eller fortsætte med at vente på noget, der allerede er stoppet. Den sidste risiko er kun observeret i det særlige simulationsforløb, ikke på hardware.
- Forslag: Vis én operatørstatus med forklaring, fx “Simulation klar – fysisk udstyr kontrolleres ikke”. Hvis efterarbejde fortsætter, skriv “Optagelse stoppet – afslutter test” og vis resten som sekundær information.
- Evidens: [første skærm](evidence/edge-control-operator-20260908/02-idle-loaded.png), [Stopped/Running](evidence/edge-control-operator-20260908/38-camera-failure-return.png).

### F02 · P1 · Et nyt scenarie viser gamle klip og overlay

- Reproduktion: Gennemfør single rider success. Vælg testmovie1 og start. Tryk Scenario source.
- UI-evidens: Det nye scenarie og nye komponentlabels stod sammen med capture-frame-2.mp4, 0 KiB, T+9.0 og overlay fra det tidligere forløb. Først reload fjernede disse gamle felter i det observerede forløb.
- Jeg forventede det valgte scenaries billede og oplysninger, men jeg så det tidligere klip blandet med den nye opsætning.
- Konsekvens: Jeg kan tro, at nye optagelser eller analyser allerede er kommet igennem. Det er ikke tydeligt, hvor aktuelt billedet er.
- Forslag: Nulstil previewets aktive klip og overlay ved nyt scenarie/ny kørsel, eller mærk dem “Tidligere kørsel” med scenarie og tidspunkt. Vis tydeligt “Testvideo”, “Aktuel test” eller “Seneste klip”.
- Evidens: [nyt scenarie med gammelt klip](evidence/edge-control-operator-20260908/11-video-selected.png), [Scenario source under opstart](evidence/edge-control-operator-20260908/13-source-while-running.png).

### F03 · P1 · Mislykket stop forklares som “Failed to fetch”

- Reproduktion: Start lokal simulation, sæt kun testbrowseren offline og klik Stop runtime. Genskab forbindelsen og gentag stop.
- UI-evidens: Rødt banner med kun Failed to fetch og luk-knap. Stop runtime forblev tilgængelig; Control connection viste Live. Nyt stop online gav til sidst Stopped.
- Jeg forventede at få at vide, om mit stop var modtaget, men jeg så en teknisk netværksfejl uden handlingsvejledning.
- Konsekvens: Jeg ved ikke, om jeg skal vente, prøve igen eller kontrollere forbindelsen. Live-indikatoren hjælper ikke med at afgøre kommandoens status. Der er ikke her bevist fuldt tab af eventforbindelsen.
- Forslag: “Stop kunne ikke bekræftes. Kontrollér forbindelsen, og prøv igen. Bear Vision kan stadig køre.” Vis kommandoens status adskilt fra seneste kendte driftsstatus og mulighed for at kopiere fejldetaljer.
- Evidens: [fejlet stop](evidence/edge-control-operator-20260908/28-offline-stop-error.png), [stop efter nyt forsøg](evidence/edge-control-operator-20260908/31-stop-after-6s.png).

### F04 · P1 · Kamerafejlen kræver teknisk fortolkning

- Reproduktion: Kør kamerafejl-fixturen til Failed, og åbn Technical details.
- UI-evidens: “Failed · Camera”, “injected camera capture failure” og “Review the technical details, then restart the runtime.” Detaljerne giver fejl-ID, Operation: Not available og Attempts: 1.
- Jeg forventede at få at vide, hvad der ikke længere virker, og hvad jeg skal gøre, men jeg så en rå fejltekst og en opfordring til at læse teknik.
- Konsekvens: Komponenten er identificeret, men jeg skal selv udlede, om nye optagelser er stoppet, om gamle klip er sikre, og hvorfor en genstart kan hjælpe. Den rå injektionstekst kommer fra testdata; fundet er UI'ens manglende forklaring omkring den.
- Forslag: “Kameraet kunne ikke optage. Der optages ikke nye klip. Kontrollér kameraets strøm og forbindelse, og prøv genstart. Kontakt support, hvis fejlen kommer igen.” Tilpas teksten til simulation og beskriv kun bevarelse af klip, når den er verificeret.
- Evidens: [kamerafejl](evidence/edge-control-operator-20260908/40-camera-final-failure.png), [tekniske detaljer](evidence/edge-control-operator-20260908/41-camera-error-filter.png).

### F05 · P1 · Stop og recovery mangler forklaring af konsekvens og udvej

- Reproduktion: Stop single rider under Camera: Capturing. Efter en vedvarende kamerafejl: forsøg at vælge andet scenarie.
- UI-evidens: Stop gav Stopping og derefter Stopped med 0 outputs; ingen forklaring på den igangværende optagelse. Efter fejl var Runtime process Exited, men scenariet var låst, og kun Restart runtime kunne vælges. Restart efterfulgt straks af Stop frigav valget.
- Jeg forventede at forstå, hvad stoppet gør ved mit klip, og kunne afslutte en fejlet test, men jeg så nul output uden forklaring og en omvej gennem genstart.
- Konsekvens: Jeg kan ikke planlægge et stop med sikker forståelse af resultatet. Ved fejl kan jeg blive ved at genstarte samme problematiske scenarie. Faktisk datatab er ikke bekræftet.
- Forslag: Forklar stop-politikken ved knappen og under Stopping. Tilbyd “Afslut fejlet test og vælg scenarie” når processen er afsluttet, med tydelig forklaring på bevaret fejl-/kliphistorik.
- Evidens: [stop under optagelse](evidence/edge-control-operator-20260908/20-mobile-stop-capturing.png), [stopresultat](evidence/edge-control-operator-20260908/21-mobile-stop-result.png), [låst fejlet kørsel](evidence/edge-control-operator-20260908/42-failure-restored.png).

### F06 · P1 · Kamera- og køstatus skjules på 320 pixels

- Reproduktion: Brug 320 × 740 viewport. Se Live/Background queue og topnavigation; kør single rider.
- UI-evidens: Vandrette scrollbars inde i kortene. Camera-trinnet og flere køtrin lå uden for det umiddelbart synlige område. Diagnostics var delvist afskåret i navigationen. Hele sidens scrollWidth var dog 320.
- Jeg forventede at kunne se kameraaktivitet uden at lede, men jeg så kun starten af statusforløbet.
- Konsekvens: Jeg kan overse den information, der fortæller, om der optages. At knapperne virker, løser ikke overbliksproblemet.
- Forslag: Brug lodrette statusrækker på små skærme, og placér “Optager/venter/stoppet” samt køtal synligt uden vandret rulning.
- Evidens: [mobilside](evidence/edge-control-operator-20260908/18-mobile-control.png), [mobil under optagelse](evidence/edge-control-operator-20260908/19-mobile-capturing.png).

### F07 · P1 · Lang opstart giver kun en voksende tæller

- Reproduktion: Vælg det eksisterende fs360 Blender-scenarie, og tryk Run scenario.
- UI-evidens: Initializing fra få sekunder til 1m 8s; Runtime process Starting, Camera Idle og Not checked. Ingen besked om ventetid, delopgave eller mulig blokering. Stop virkede senere.
- Jeg forventede at kunne afgøre, om jeg skulle vente eller gøre noget, men jeg så kun Initializing og tid, der gik.
- Konsekvens: Jeg kan afbryde en sund opstart eller vente unødigt på et problem. Testen afgør ikke, hvilken af de to situationer dette scenarie var i.
- Forslag: Vis hvad der forberedes, og hvornår status sidst ændrede sig. Giv realistisk ventetidsvejledning og en klar handling ved usædvanligt lang opstart; undgå opdigtede procenttal.
- Evidens: [22 sekunder](evidence/edge-control-operator-20260908/25-blender-progress.png), [1m 8s ved stopforsøg](evidence/edge-control-operator-20260908/30-retry-stop-online.png).

### F08 · P2 · Scenarievalg og hjælpetekst taler udviklersprog

- Reproduktion: Åbn scenarielisten og læs previewets labels.
- UI-evidens: Navne som “wakeboard fs360 60fps blender regression”, “yolo”, “frames: synthetic”, “bear_tag”, “Kalman + RTS” og “Butterworth camera crop”. Ingen synlig, kort beskrivelse af formål eller forventet resultat.
- Jeg forventede at vælge en egnet prøve ud fra dens formål, men jeg så tekniske navne, jeg måtte gætte betydningen af.
- Konsekvens: Jeg kan vælge en langsom eller uegnet prøve. Single rider success var nemmest at afkode.
- Forslag: Giv scenarier formålsnavne, fx “Én kører – normal prøve uden video”, med forventet varighed/resultat. Flyt algoritme- og komponentnavne til Technical details.
- Evidens: [valg og labels](evidence/edge-control-operator-20260908/02-idle-loaded.png), [Blender valgt](evidence/edge-control-operator-20260908/23-blender-selected.png).

### F09 · P2 · Afsluttet test har et uklart klipresultat

- Reproduktion: Lad single rider success afslutte, og sammenhold hovedstatus, kø, preview og historik.
- UI-evidens: Idle, køens 0 completed, Recent runs med 1 output og Extracted clip på 0 KiB. En faktisk afspillelig klipfil blev ikke bekræftet.
- Jeg forventede en enkel afslutningsbesked, men jeg så både nul færdige klip, ét output og et tomt klip.
- Konsekvens: Jeg kan forveksle et syntetisk testresultat med en rigtig video eller tro, at resultatet er forsvundet. Nulbyte-output er ikke i sig selv dokumenteret som en fejl i denne syntetiske simulation.
- Forslag: Vis “Test gennemført – 1 syntetisk klipresultat, ingen videofil”. Mærk køtal “Aktuel kørsel”, og vis seneste resultat tæt ved hovedstatus.
- Evidens: [normal afslutning](evidence/edge-control-operator-20260908/10-normal-run-final.png).

### F10 · P2 · Supportevidens er vanskelig at dele og mister historik ved reload

- Reproduktion: Åbn Diagnostics efter drift/fejl. Se tidsangivelser, vælg Error, og genindlæs.
- UI-evidens: LIVE/T+ frem for klokkeslæt på mange hændelser; ingen synlig kopi-/eksporthandling. Ved kamerafejl fjernede Error-filteret stacktrace, som stod som Info log, men viste fortsat almindelige hændelser. Reload bevarede fejlkortet, mens Diagnostics blev 0/0.
- Jeg forventede at kunne give support et samlet hændelsesforløb, men jeg så en teknisk liste, som ændrede sig med filter og reload.
- Konsekvens: Jeg må selv udvælge tekst eller skærmbilleder, og support kan mangle rækkefølge og tidspunkt. Deling med faktisk support blev ikke udført.
- Forslag: Tilføj “Kopiér oplysninger til support” med tidspunkt, scenarie, seneste handling, resultat og fejl-ID. Bevar relevante hændelser efter reload, og gør filterets betydning tydelig.
- Evidens: [Diagnostics efter stop](evidence/edge-control-operator-20260908/22-diagnostics-stop.png), [Error-filter](evidence/edge-control-operator-20260908/41-camera-error-filter.png), [efter reload](evidence/edge-control-operator-20260908/42-failure-restored.png).

## Fejlmeddelelser vurderet mod operatørens spørgsmål

| Spørgsmål | Fejlet stop: Failed to fetch | Kamerafejl |
| --- | --- | --- |
| Hvad er galt? | Nej, kun teknisk fejltekst. | Delvist: Camera og rå fejloplysning. |
| Hvad er påvirket? | Stophandlingen nævnes ikke. | Kamera nævnes; konsekvens for optagelser/klip mangler. |
| Kan driften fortsætte? | Ikke forklaret; Live står fortsat. | Exited/Failed skal fortolkes; ingen almindelig sætning om driften. |
| Hvad skal jeg gøre? | Ingen vejledning. | Genstart nævnes, men efter læsning af tekniske detaljer. |
| Kan support få detaljer uden at jeg forstår dem? | Intet synligt supportudtræk i banneret. | Sammenklappede detaljer med fejl-ID; manuel kopiering/skærmbillede mulig, ingen observeret eksportknap. |

## Scorer

| Område | Score | Kort begrundelse |
| --- | --- | --- |
| Overblik | 5/10 | Relevant information findes, men modstridende status og skjulte mobilfelter kræver aktiv fortolkning. |
| Start og readiness | 6/10 | Start er let og kvitteres. Readiness i simulation og lang opstart er uklare. Hardware-readiness er ikke vurderet. |
| Forståelse under drift | 4/10 | Capturing er nyttigt; gamle overlay/klip og tekniske pipeline-begreber svækker tilliden. |
| Stop og recovery | 5/10 | Stop virkede, også under simuleret optagelse. Klipkonsekvens og vej ud af en afsluttet fejl er uklare. |
| Fejlmeddelelser og næste handling | 3/10 | Fejl er synlige, men forklaring af påvirkning og næste handling er utilstrækkelig. |
| Samlet | 5/10 | Afrundet helhedsvurdering af det afprøvede simulationsscope; ikke en måling af teknisk stabilitet eller fysisk drift. |

## De tre vigtigste ændringer

1. Saml tilstand, readiness og næste handling i et tydeligt operatøroverblik. Skeln mellem klar, optager, afslutter og stoppet; vis kamera og kø uden vandret rulning.
2. Gør stop- og fejlbeskeder handlingsanvisende: modtaget/ikke bekræftet, påvirket funktion, hvad der sker med klip, og konkret vej videre. Tilbyd afslutning af fejlet test uden omvejen gennem genstart.
3. Adskil aktuelle billeder og resultater fra historik. Fjern gamle overlay ved scenarieskift, og mærk testvideo, syntetiske resultater og seneste klip tydeligt med aktualitet.

## Ikke testet eller ikke bekræftet

- Fysisk kamera, BLE/BearTag, ægte driftsstart, hardware-readiness, blokerende hardwarechecks og advarselskvitteringer. Hardwaretilstand blev ikke valgt.
- En vellykket readiness-kontrol gennem simulationens UI; Not checked stod fortsat.
- Vedvarende samtidig optagelse og Processing/Packaging/Uploading, voksende kø eller stop med et aktivt baggrundsjob. Trinene blev set som labels/hændelser, ikke som et tilstrækkeligt langt operatørforløb.
- Isoleret klipfejl og Retry clip job. Det gyldige storage_upload-forsøg gav Completed uden fejl. Teknisk kontrol viste, at portalens lokale kø erstatter den simulerede fejlede publiceringsadapter; dette forklarer testbegrænsningen, ikke UI-forståeligheden.
- Fuld afbrydelse af en allerede åben eventforbindelse. Browserens offline-test bekræftede en fejlet stopkommando, ikke alle typer netværksudfald.
- Force stop, konsekvenser for ventende klip og genstart af en kørende backendserver. Force stop blev ikke synligt nødvendigt i de afprøvede forløb.
- Fuldført Blender-/YOLO-videobehandling, korrekt outputkvalitet, faktisk afspilning af genererede klip eller fysisk databevarelse.
- Ægte mobiltelefon/touch, andre browsere, tastatur-/skærmlæsergennemgang, længere tids drift og flere samtidige operatører.
- Faktisk deling af fejl med support eller test med mennesker. Ingen beskeder blev sendt.

## Testmiljøets sluttilstand

Sidste færdigindlæste UI-observation kl. 21:54:53 viste Simulation, single rider success, Idle, Runtime process Idle, Camera Idle, nul køaktivitet og aktiv Run scenario. Testhistorikken var bevaret i det isolerede testmiljø. Readiness stod stadig Not checked.

Serveren, den isolerede Chrome-browser og deres identificerede underprocesser blev derefter stoppet. Kontrol kl. 21:54:58 fandt ingen resterende egne processer og ingen lyttere på 4310 eller browserens testport 9431. Ingen Python-underproces var tilbage i serverens procestræ før nedlukning.

De to midlertidige scenarier er fjernet fra `specs/scenarios`. Reproduktionskopier, skærmbilleder og oprydningskontrol er bevaret i evidensmappen. Ignorerede build-/testartefakter, isoleret kø, browserprofil og logs er efterladt under `apps/edge-control/dist`, `temp/operator-exploration-20260908` og simulationens egne mapper under `temp/captures`; de kører ikke. Applikationskoden er uændret. Eksisterende uvedkommende filer blev ikke ændret.

- [Sidste UI-skærm](evidence/edge-control-operator-20260908/45-final-loaded-idle.png).
- [Sidste UI-tekst](evidence/edge-control-operator-20260908/45-final-loaded-idle.txt).
- [Verificeret oprydning](evidence/edge-control-operator-20260908/cleanup.json).
