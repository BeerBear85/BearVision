# Bear Vision Video Server — high-level testspecifikation

Status: Levende acceptspecifikation for operatørvendt adfærd.

## Formål

Dette dokument beskriver den forventede, observerbare opførsel for Bear Vision
Video Server og Server Control. Det er en stabil baseline for manuelle,
exploratory og automatiserede tests. Det beskriver hvad operatøren skal kunne
opnå, ikke en bestemt skærmstruktur eller teknisk implementering.

En dateret testrapport dokumenterer, hvad der faktisk blev observeret ved en
konkret testkørsel. Den skal referere til scenarie-ID'erne i dette dokument og
må ikke bruges som erstatning for specifikationen.

## Scope

Specifikationen dækker:

- serverens drifts- og klipoverblik;
- modtagelse, behandling og bevaring af klip;
- automatisk og manuel rider assignment;
- korrekt håndtering af unresolved-klip;
- bruger- og BearTag-historik;
- konsekvensvisning og genberegning efter historikkorrektioner;
- fejl, stilstand og recovery;
- operatørfeedback, sikkerhedsgrænser og adgang til medier.

Følgende kræver særskilte testmissioner og er ikke generelle acceptkrav her:

- kalibrering af scoringalgoritmens faglige nøjagtighed;
- Edge-enhedens optagelsesadfærd;
- belastnings-, penetration- og disaster-recovery-test;
- Box eller anden storageproviders interne brugerflade.

## Grundlæggende acceptprincipper

- Operatøren skal kunne forstå status og næste handling gennem den synlige UI
  uden at læse kode, rå logs eller interne fejlobjekter.
- `Unresolved` er et gyldigt resultat, når grundlaget for en sikker assignment
  mangler. Det må ikke fremstilles som et mistet klip.
- Et problem med ét klip må ikke fremstilles som et generelt serverstop.
- Modtagne klip og deres identitet skal bevares gennem fejl, korrektioner,
  requeue og genberegning.
- Handlinger skal have synlig feedback og må ikke efterlade operatøren i tvivl
  om, hvorvidt ændringen blev modtaget eller gennemført.
- Ikke-slettende handlinger skal kunne udføres uden ekstra bekræftelse.
  Sletning skal kræve en tydelig bekræftelse, der navngiver mål og konsekvens.

## Forventet opførsel

### BVS-01 — Første driftsoverblik

En operatør skal inden 30 sekunder kunne se:

- om serveren kører normalt, er forsinket eller kræver handling;
- tidspunktet for senest modtagne klip;
- samlet antal relevante klip og fordelingen mellem assigned, unresolved,
  processing og failed;
- om viste data er aktuelle.

Status må ikke modsige sig selv, eksempelvis ved at vise serveren som normal
samtidig med, at forbindelsen er afbrudt og tallene er forældede.

### BVS-02 — Modtagelse og behandling af et nyt klip

Et komplet, nyt klip skal registreres én gang og bevæge sig gennem forståelige
tilstande frem mod assigned, unresolved eller failed. UI'en skal opdatere
status uden at kræve manuel genstart.

Et ufuldstændigt klip må ikke behandles som færdigt. Gentagen registrering af
samme job må ikke skabe dubletter eller flere modstridende resultater.

### BVS-03 — Automatisk rider assignment

Når et klip tildeles automatisk, skal operatøren kunne se:

- hvilken bruger klippet er tildelt;
- hvilket BearTag- og tidsgrundlag der blev anvendt;
- en forståelig begrundelse og tilstrækkelig score-evidens til at vurdere
  resultatet;
- at klippet fortsat kan åbnes og afspilles.

Brugerens stabile identitet må ikke afhænge af en emailadresse, som senere kan
ændres.

### BVS-04 — Unresolved-klip

Et klip uden tilstrækkeligt assignmentgrundlag skal:

- være tydeligt markeret `Unresolved` og forblive synligt og afspilleligt;
- forklare, hvorfor en sikker assignment ikke kunne foretages;
- kunne forblive unresolved, tildeles manuelt eller requeues efter behov;
- skelnes tydeligt fra teknisk fejlede eller mistede klip.

### BVS-05 — Søgning, filtrering og medieadgang

Operatøren skal kunne finde et klip via relevante status-, bruger- og
tekstfiltre, også når datamængden kræver flere sider. Filtrering og refresh må
ikke ændre klippets status eller assignment.

Thumbnail og video skal tilhøre det valgte klip. Videoen skal kunne afspilles
og søges i uden at udlevere en anden brugers medie eller acceptere et
uverificeret medie som det korrekte klip.

### BVS-06 — Manuel omfordeling

Et processed eller unresolved klip skal kunne tildeles en anden bruger med en
kort begrundelse. Efter handlingen skal den nye bruger og den manuelle kilde
være synlige med det samme, mens klippets stabile identitet og medie bevares.

Den tidligere assignment og begrundelsen for korrektionen skal kunne spores.
Handlingen er ikke en sletning og skal ikke kræve ekstra bekræftelse.

### BVS-07 — Redigering af brugerdata

Operatøren skal kunne rette en brugers navn og email uden at oprette en ny
identitet eller ændre brugerens UUID. Eksisterende klip og assignments skal
fortsat pege på den samme bruger.

Før- og efterværdier, tidspunkt og begrundelse skal kunne spores. Valideringsfejl
skal forklare problemet og den mulige rettelse i operatørsprog.

### BVS-08 — BearTag-historik

Operatøren skal kunne oprette, flytte og opdele et BearTag-ejerskab i
tidsafgrænsede intervaller. Tidszone og grænser skal være forståelige.

Historikken må ikke acceptere overlap eller huller, når et eksisterende
interval erstattes. En konflikt skal vises med den berørte BearTag, bruger og
tidsperiode samt en konkret rettemulighed frem for rå valideringsdetaljer.

### BVS-09 — Konsekvensvisning og genberegning

Før en BearTag-historikændring anvendes, skal operatøren kunne se alle berørte
klip med nuværende og forventet status og bruger. Oversigten skal skelne mellem
ændrede, uændrede og unresolved resultater.

Når ændringen anvendes, skal det samme afgrænsede sæt genberegnes som én
sammenhængende handling. Resultatet skal vise før og efter. Manuelle
assignments skal bevares, medmindre operatøren udtrykkeligt vælger at
overskrive dem.

### BVS-10 — Requeue

Et klip skal kunne requeues uden at miste medie, manifest, stabil jobidentitet
eller revisionsspor. Operatøren skal se, at handlingen er modtaget, hvilken
tilstand klippet går til, og hvilket nyt terminalt resultat der opstår.

Requeue er ikke en sletning og skal ikke kræve ekstra bekræftelse.

### BVS-11 — Fejl, stilstand og recovery

Når serveren eller en nødvendig proces ikke kan nås, skal UI'en tydeligt vise:

- at de viste data kan være forældede;
- hvad der er påvirket;
- om klipbehandling fortsætter eller er stoppet;
- hvad operatøren kan gøre nu.

Efter recovery skal aktuelle tal og status gendannes uden datatab eller
modstridende rester fra fejltilstanden.

### BVS-12 — Bevaring og sporbarhed

Et modtaget klip skal kunne spores gennem processing, assignment, unresolved,
failed, requeue og korrektioner. Korrektioner må kun ændre det tilsigtede
resultat; mediet og urelaterede klip skal forblive uændrede.

Historiske værdier, manuelle beslutninger og væsentlige ændringer skal have et
revisionsspor, som support kan forbinde med bruger, tidspunkt og årsag.

### BVS-13 — Sprog og feedback

Server Control skal bruge konsistent engelsk UI-tekst og forklare opgaver,
konsekvenser og næste handling i almindeligt sprog. Interne komponentnavne,
storageproviders, stack traces, valideringsobjekter og dokumentations-URL'er må
ikke være nødvendige for at gennemføre en operatøropgave.

Langsomme handlinger skal vise en tydelig igangværende tilstand og forhindre
utilsigtede dubletter. Resultatet af en handling skal forblive synligt længe
nok til, at operatøren kan forstå det.

### BVS-14 — Adgangsgrænser

Server Control er en lokal administrationsflade og må ikke eksponeres som en
generel LAN-tjeneste. En separat brugerrettet LAN-API skal være read-only og må
kun udlevere medier, der tilhører den pågældende bruger.

## Testudførelse

En testkørsel skal som minimum angive:

- testet commit, konfiguration, miljø og datakilder;
- anvendte testbrugere, BearTags og klip;
- hvilke BVS-scenarier der blev gennemført;
- resultatet `Bestået`, `Delvist bestået`, `Fejlet` eller `Ikke testet`;
- synlig evidens og forskellen mellem forventet og observeret opførsel;
- miljøets sluttilstand og eventuelle data, der bevidst blev bevaret.

UI-adfærd skal først vurderes gennem den synlige brugerflade. Kode, API'er og
automatiske tests må bruges bagefter til testopsætning og kontrol, men kan ikke
bevise, at operatørens oplevelse er forståelig.

Brug lokal simulation eller et udtrykkeligt godkendt testmiljø. En testmission
giver ikke i sig selv tilladelse til at ændre eller slette produktionsdata.

## Rapportering og prioritet

Fund prioriteres efter konsekvens:

- P0: risiko for tab, utilsigtet adgang eller korruption af klip eller identitet;
- P1: en central operatøropgave er blokeret eller giver et forkert resultat;
- P2: drift eller konsekvens kan misforstås, eller næste handling er uklar;
- P3: mindre friktion, sprog- eller præsentationsproblem.

En rapport skal skelne mellem observerede fund, udledte risici og ikke testede
forhold. Hvert væsentligt fund skal knyttes til et BVS-scenarie og beskrive
situation, forventning, observation, konsekvens og korte reproduktionstrin.

## Relaterede dokumenter

- `apps/server-control/README.md`: produktets aktuelle ansvar og lokale start.
- `specs/contracts/server-control-api.md`: den lokale administrationskontrakt.
- `specs/contracts/server-job-v1.md`: jobpakke, status og mediebevaring.
- `docs/testing/server-control-exploratory-operator-report-20260911.md`:
  historisk testresultat, ikke normativ forventning.
