# Edge Control – exploratory usability-test som kabelparkoperatør

## Resumé

Edge Control kan bruges til det normale simulationsforløb uden teknisk hjælp. Portalen gør det tydeligt, om Bear Vision er klar, starter, optager, kører normalt eller kræver handling. Kameraaktivitet og baggrundsarbejde er adskilt på en måde, der giver mening for en operatør, og en aktiv kørsel blev gendannet korrekt efter reload.

Samlet usability-score: **7/10**. Der blev ikke observeret P0-problemer. De største risici er, at en sund, men langsom start kan ligne en fastlåst start, og at et gennemført simulationsresultat samtidig kan fremstå som et 0 KiB-klip, der ikke kan afspilles. Hardware-readiness og en samlet kamerafejl gav derimod konkrete og forståelige næste handlinger.

Dette er en simuleret personas vurdering, ikke et studie med faktiske kabelparkoperatører.

## Testmiljø, tidspunkt og begrænsninger

- Testet 9. september 2026 cirka 19:34–19:48, Europe/Copenhagen.
- Repository: `D:/Repositories/BearVision_2`, branch `master`, commit `4c46400f17a20ca1e57de835528ee8f61f6dde09` (`Enable recovery of terminal failed runs and media errors`).
- Lokal redeployment efter `apps/edge-control/README.md`: `corepack pnpm install --frozen-lockfile`, `corepack pnpm build` og `node server/server.mjs`. Build lykkedes, og portalen svarede HTTP 200 på `http://localhost:4310`.
- Portalen blev betjent i Codex' lokale browser ved normal desktopstørrelse gennem synlige labels, knapper, statusfelter og scenarievælger.
- Kode og terminal blev kun brugt til build, serverstyring, miljøkontrol og midlertidig fejlinjektion. Det tæller ikke som bevis for, at UI'en er forståelig.
- Simulation var hovedmiljøet. Hardware-readiness blev kørt; den fandt syv BearTag-advertisements, men ingen GoPro. Fysisk Bear Vision-drift blev ikke startet.
- To midlertidige scenarier blev brugt til kamerafejl og uploadfejl. De blev fjernet igen. Applikationskode blev ikke ændret.
- Der blev ikke testet mobilviewport, touch, andre browsere, langvarig drift, strømsvigt eller flere samtidige operatører.

“Observeret” betyder synligt i portalen efter en faktisk UI-handling. “Udledt” er operatørkonsekvensen vurderet ud fra observationen. “Ikke testet” bruges, når miljøet ikke gav pålideligt UI-bevis.

## Gennemførte brugerforløb

| Valg og forventning før handling | Faktisk synligt resultat | Evidensstatus og konsekvens |
| --- | --- | --- |
| Åbn portalen; jeg forventede straks at kunne se klar/kører/problem/næste handling. | Den bevarede hardwaretilstand viste “ACTION REQUIRED”, “Check the hardware before starting”, Camera Idle, Connection Live og deaktiveret Start hardware. | Observeret. Første blik var entydigt; det var også klart, at en handling var nødvendig. |
| Vælg Simulation; jeg forventede, at hardwarekravet forsvandt. | Efter cirka 0,5 sekund stod “Simulation is ready to run” og “Physical equipment is not checked in simulation mode”. Run scenario blev aktiv. | Observeret. Operatøren kan skelne test fra rigtig drift. |
| Vælg et normalt scenarie; jeg forventede et valg efter formål. | “single rider success” var forståeligt. Flere andre navne indeholdt `yolo`, `60fps`, `blender regression`, Kalman/RTS og Butterworth. | Observeret. Ét sikkert valg var let; resten kræver teknisk gætteri. |
| Dobbeltklik Run scenario; jeg forventede højst én kørsel. | Knappen forsvandt/deaktiveredes, status skiftede til Starting, og kun én ny kørsel kom i Recent runs. | Observeret i dette forsøg. Ingen synlig dobbeltstart. |
| Følg normal drift; jeg forventede at forstå kamera og kliparbejde. | “BearVision is recording riders”, Camera Capturing og teksten “New clips are being captured while earlier clips can finish in the background”. Køen viste aktive/færdige klip og Current clip job. | Observeret. Liveoptagelse og baggrundsbehandling er forståeligt adskilt. |
| Genindlæs siden under aktiv optagelse; jeg forventede samme kørsel tilbage. | Efter reload stod samme scenarie låst som Recording, Camera Capturing, Background 1 active, Connection Live og Stop runtime. | Observeret. Aktiv drift blev gendannet korrekt. |
| Forlad portalen kort og vend tilbage; jeg forventede at forstå, hvad der var sket imens. | Ved tilbagevenden var kørslen færdig; Simulation ready og en ny Completed-række med 1 output og 0 failures var synlig. | Observeret. Historikken giver et brugbart svar efter en afbrydelse. |
| Start et stort Blender-scenarie; jeg forventede løbende fremdrift eller ventetid. | “BearVision is starting” og “Preparing the selected runtime. No action is needed yet” stod i cirka 30 sekunder med kun en stigende sekindtæller. Derefter blev Running observeret. | Observeret. Systemet arbejdede, men operatøren kan ikke skelne sund ventetid fra fastlåsning. |
| Stop det lange scenarie under Running; jeg forventede tydelig kvittering og klippets skæbne. | Portalen sprang hurtigt til Simulation ready. Recent runs viste Stopped, 1 output og 0 failures. | Observeret. Stop lykkedes, men UI sagde ikke, hvad der var afsluttet, bevaret eller kasseret. |
| Kør hardware-readiness uden GoPro; jeg forventede en konkret blokering. | “Hardware is not ready”, “BLOCKING ISSUES 1” og “GoPro: A scan timed out without finding a device”. UI bad om at tilslutte/tænde GoPro via USB, lukke andre preview-forbrugere, kontrollere USB-netværk og køre readiness igen. Start hardware var deaktiveret. | Observeret. Problem, påvirkning og næste handling var tydelige. |
| Kør et simuleret kameraudfald; jeg forventede forskel på samlet driftsfejl og klipfejl. | “BearVision needs attention”, “The simulated camera could not record” og “No new test clips are being captured. Restart the runtime to continue the test.” | Observeret. Dette var en fejl i hele optagelsen, ikke ét klip. |
| Åbn Technical details; jeg forventede delbare supportdata uden at skulle forstå dem. | Failure-id, rå fejl `injected camera capture failure`, tidspunkt og Attempts 1 var sekundært placeret. | Observeret. Data kan deles manuelt; operatøren behøver dem ikke for at vælge handling. |
| Vælg End failed run; jeg forventede at kunne forlade fejlen uden at gentage scenariet. | Efter cirka 0,5 sekund blev setup frigivet, Simulation ready kom tilbage, og historikken bevarede Failed, 0 outputs og 1 failure. | Observeret. Recovery er logisk og bevarer evidensen. |
| Kør et uploadfejlscenarie; jeg forventede en isoleret klipfejl. | Portalen viste Completed, 1 output og 0 failures. Upload var deaktiveret i den aktive konfiguration. | Observeret resultat, men isoleret klipfejl er ikke testet: miljøet frembragte ikke den tilsigtede UI-tilstand. |

## Det fungerede godt

- Hovedstatussen svarer ved første blik på, om Bear Vision er klar, kører eller kræver handling.
- Mode vises både i topstatus og kontrolområdet, og simulation forklarer eksplicit, at fysisk udstyr ikke kontrolleres.
- Start låser scenarie og mode og modstod det afprøvede dobbeltklik.
- Kameraaktivitet og baggrundsklip er adskilt, og teksten forklarer, at optagelse kan fortsætte, mens tidligere klip behandles.
- Reload under Recording gendannede kørsel, stopmulighed, kameraaktivitet og køstatus.
- Hardware-readiness blokerede start korrekt og gav en konkret GoPro-handling.
- Kamerafejlen forklarede både hvad der var galt, hvad der var påvirket og hvordan testen kunne fortsætte.
- Technical details var sekundære, og “End failed run” gav en direkte vej tilbage til idle uden at gentage fejlen.

## Prioriterede findings

Der blev ikke observeret P0-findings.

### F01 · P1 · Completed-resultat ligner et ubrugeligt klip

- Situation og reproduktion: Kør `single rider success` til afslutning og se Extracted clip + live overlay.
- Konkret UI-evidens: Recent runs viste Completed, 1 output og 0 failures. Medieområdet viste samtidig “Unable to play media” og `capture-frame-2.mp4 · 0 KiB`.
- Jeg forventede et afspilleligt eller tydeligt markeret syntetisk resultat, men jeg så et gennemført resultat med et klip på 0 KiB, som browseren ikke kunne afspille.
- Konsekvens for operatøren: Jeg kan ikke afgøre, om Bear Vision har leveret et brugbart klip, eller om kun simulationens status er lykkedes. Det svækker tilliden til både resultat og fejlstatus.
- Forslag: Vis “Simulation uden videofil” som normal, neutral tilstand, når scenariet kun har syntetiske data. Vis ikke et 0 KiB-klip som et færdigt medieoutput. Hvis et klip forventes, skal tilstanden i stedet være en afgrænset resultatfejl med konkret retry.
- Evidensstatus: Observeret i portalen; faktisk filindhold blev ikke testet.

### F02 · P1 · Lang opstart mangler forventet varighed og fremdrift

- Situation og reproduktion: Start `wakeboard two riders 60fps blender regression · Blender`.
- Konkret UI-evidens: Starting stod i cirka 30 sekunder. Teksten var kun “Preparing the selected runtime. No action is needed yet”, Camera Idle, 0 active/queued og en stigende tid. Scenariet nåede senere Running.
- Jeg forventede et forventet tidsinterval eller et konkret forberedelsestrin, men jeg så kun, at jeg fortsat skulle vente.
- Konsekvens for operatøren: Jeg kan tro, at systemet er fastlåst, og stoppe en sund start. Omvendt kan jeg vente for længe på en reel fejl.
- Forslag: Vis forventet starttid for det valgte scenarie og seneste konkrete fremdrift, for eksempel “Indlæser video — normalt 20–40 sekunder”. Efter intervallet skal UI foreslå en handling uden falske procenttal.
- Evidensstatus: Observeret; scenariet gik senere til Running.

### F03 · P2 · Stop forklarer ikke, hvad der sker med igangværende arbejde

- Situation og reproduktion: Start det lange to-rider-scenarie, vent til Running, og vælg Stop runtime.
- Konkret UI-evidens: Status sprang hurtigt til Simulation ready. Recent runs viste Stopped, 1 output og 0 failures, men ingen forklaring af det resterende kliparbejde.
- Jeg forventede en kvittering som “Stop modtaget” og en forklaring af, hvad der færdiggøres eller kasseres, men jeg så kun slutstatus og outputtal.
- Konsekvens for operatøren: Jeg ved ikke, om det er sikkert at slukke udstyr, eller om et aktivt klip stadig færdiggøres. Faktisk datatab blev ikke observeret.
- Forslag: Vis en kort Stopping-tilstand og afslutningsresultat: “Kamera stoppet; 1 klip bevaret; ingen klip venter” eller den faktiske politik.
- Evidensstatus: Observeret; konsekvensen er udledt.

### F04 · P2 · Scenarievalg er skrevet til udviklere

- Situation og reproduktion: Åbn Scenario-listen og sammenlign valgmuligheder og preview.
- Konkret UI-evidens: Navne som “wakeboard testmovie1 yolo” og “wakeboard two riders 60fps blender regression · Blender”; preview viste Kalman + RTS og Butterworth.
- Jeg forventede valg efter operatørformål, men jeg så implementerings- og regressionstermer.
- Konsekvens for operatøren: Jeg kan vælge et langsomt eller forkert scenarie uden at forstå forskellen.
- Forslag: Vis et operatørnavn, formål, forventet varighed og forventet resultat. Flyt filnavn og algoritmer til Technical details.
- Evidensstatus: Observeret.

### F05 · P2 · Reload kan kort vise Ready og Reconnecting samtidig

- Situation og reproduktion: Genindlæs portalen tæt på afslutningen af en kørsel.
- Konkret UI-evidens: I ét reload-forsøg stod “Simulation is ready to run” samtidig med Connection Reconnecting, tom scenarieværdi og “No completed runs yet”. Efter under ét sekund var Connection Live, scenarie og historik gendannet. Et separat reload under aktiv Recording gendannede korrekt tilstand med det samme.
- Jeg forventede en neutral “Indlæser seneste tilstand”, men jeg så kort en autoritativ Ready-status før data var gendannet.
- Konsekvens for operatøren: Ved et hurtigt blik kan jeg tro, at alt er klar, selv om forbindelsen og historikken endnu ikke er afklaret. Startknappen var deaktiveret i mellemtilstanden.
- Forslag: Vis Loading/Reconnecting som hovedstatus, indtil første komplette snapshot er modtaget. Undgå Ready og tom historik i den periode.
- Evidensstatus: Observeret én gang; ikke reproduceret ved det senere aktive reload.

### F06 · P2 · Supportevidens kan aflæses, men ikke deles med ét tryk

- Situation og reproduktion: Fremprovokér kamerafejl, åbn Technical details og Diagnostics.
- Konkret UI-evidens: Fejlkortet viste failure-id, rå fejl og attempts. Diagnostics viste generiske rækker som “Readiness updated” og “Runtime mode changed”. Der var ingen synlig Copy support details-handling på fejlkortet.
- Jeg forventede at kunne kopiere en samlet hændelse til support, men jeg så data, der skulle markeres og samles manuelt.
- Konsekvens for operatøren: Jeg kan stadig hjælpe support, men risikoen for at udelade id eller tidspunkt er højere.
- Forslag: Tilføj “Kopiér supportdetaljer” på hvert persistent fejlkort med tidspunkt, failure-id, operation/job-id og rå fejl. Behold detaljer skjult som nu.
- Evidensstatus: Observeret.

## Fejlmeddelelser og næste handling

| Fejltilstand | Hvad er galt? | Hvad er påvirket? | Kan driften fortsætte? | Konkret operatørhandling | Supportdetaljer |
| --- | --- | --- | --- | --- | --- |
| Hardware-readiness uden GoPro | Ja: scan fandt ingen GoPro. | Hardwarestart er blokeret. | Nej, ikke før blokeringen er løst. | Tænd/tilslut GoPro via USB, luk andre preview-forbrugere, kontrollér USB-netværk, og kør readiness igen. | Passed-listen viste bl.a. runtime, model, medieværktøjer, storage og BearTag-scan. |
| Simuleret kamerafejl | Ja: kameraet kunne ikke optage. | Ingen nye testklip optages. | Ja, efter Restart runtime, eller testen kan afsluttes med End failed run. | Restart runtime eller End failed run. | Failure-id, tidspunkt, rå fejl og attempts under Technical details. |
| Forsøgt isoleret uploadfejl | Nej: portalen viste Completed og 0 failures. | Ikke afklaret. | Ikke afklaret. | Ingen fejlhandling blev vist. | Ikke testet pålideligt, fordi upload var deaktiveret. |

Fejlmeddelelserne for de to observerede reelle blokeringer besvarede de fire vigtigste spørgsmål: hvad der er galt, hvad der er påvirket, om man kan fortsætte, og hvad man konkret skal gøre. Tekniske detaljer var ikke nødvendige for at vælge handling. En isoleret fejl i ét klip blev ikke fremprovokeret pålideligt og må derfor testes særskilt i et miljø med aktiv klipkø/upload.

## Scorer

| Område | Score | Begrundelse |
| --- | --- | --- |
| Overblik | 8/10 | Hovedstatus, mode, kamera, kø og forbindelse er tydelige. En kort reload-mellemtilstand kan modsige sig selv. |
| Start og readiness | 7/10 | Simulation og hardwareblokering er klare; tekniske scenarienavne og 30 sekunders uforklaret start trækker ned. |
| Forståelse under drift | 8/10 | Recording, kamera og baggrundsarbejde er godt forklaret, og aktiv drift overlever reload. 0 KiB-resultatet svækker tilliden. |
| Stop og recovery | 7/10 | Stop og End failed run virker. Det mangler at fremgå, hvad der sker med igangværende klip. |
| Fejlmeddelelser og næste handling | 8/10 | Hardware- og kamerafejl er konkrete og handlingsorienterede. Isoleret klipfejl blev ikke testet, og supportkopiering er manuel. |
| Samlet | 7/10 | Kerneforløbet er brugbart, men resultattillid og ventetid ved start er fortsat væsentlige operatørrisici. |

## De tre vigtigste anbefalede ændringer

1. Gør klipresultatet entydigt: vis ikke Completed sammen med et 0 KiB/ikke-afspilleligt klip uden at forklare, at scenariet bevidst ikke producerer video.
2. Giv langsom opstart et forventet tidsinterval og konkrete fremdriftstrin.
3. Forklar ved stop, hvad der sker med kamera, ventende klip og allerede producerede outputs.

## Ikke testet eller ikke bekræftet

- En isoleret fejl i ét klip, clip retry og fortsat drift med andre klip. Uploadfejlsforsøget gav ikke en fejl i den aktive konfiguration.
- Hardwarestart, rigtig GoPro-preview, fysisk optagelse og stop af fysisk hardware.
- Advarselskvittering ved ikke-kritiske readiness-advarsler.
- Force stop efter udløbet graceful-stop-timeout.
- Kontrolserverafbrydelse midt i en stophandling og recovery efter længere forbindelsestab.
- Stop mens et klip var synligt i Processing, Packaging eller Uploading. Stop blev observeret under Running med et produceret output.
- Faktisk indhold og kvalitet i video-/trackingfiler; vurderingen er af portalens synlige kommunikation.
- Clipboard-indhold og faktisk deling med support.
- Mobil/touch, andre browsere, skærmlæser, langvarig drift, samtidige operatører og strømudfald.

## Testmiljøets sluttilstand

Sidste UI-observation før nedlukning viste Simulation ready, `single rider success`, Camera Idle, Background 0 active/queued, Connection Live og aktiv Run scenario. De midlertidige fejlscenarier var ikke længere i scenarielisten. Recent runs bevarede både Completed-, Stopped- og Failed-evidens fra testen.

Efter rapporten blev skrevet, blev browserfanen lukket og alle lokale Edge Control-processer startet til testen stoppet. Den endelige tekniske kontrol fandt ingen lytter på port 4310. Der blev ikke implementeret rettelser.
