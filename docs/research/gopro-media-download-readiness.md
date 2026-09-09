# GoPro media-download readiness efter stop

Dato: 2026-09-04

## Konklusion

En HTTP 503 fra GoProens fil-URL skal behandles som en midlertidig fejl, men GoPro
dokumenterer ikke 503 som et specifikt signal om, at en optagelse stadig
finaliseres. Det er derfor en sandsynlig forklaring i dette forløb, ikke en
dokumenteret GoPro-kontrakt.

Den robuste løsning er en lagdelt readiness-kontrol:

1. Vent på `READY=true`, `BUSY=false` og `ENCODING=false`.
2. Find og fasthold den præcise nye `camera_file`; vælg ikke "nyeste fil" igen
   under retry.
3. Poll filressourcen med `HEAD` og kræv `200`, positiv `Content-Length` og samme
   længde i en kort, tidsbaseret stabilitetsperiode.
4. Download til en `.part`-fil med begrænset retry på 503 og transportfejl.
5. Kontrollér byteantal mod filressourcens længde igen efter download, kør den
   eksisterende medieprobe, og omdøb først derefter atomisk til slutnavnet.

Forslaget om 2 Hz og tre stabile målinger er en fornuftig del af løsningen, men
det er ikke tilstrækkeligt alene: tre målinger ved 2 Hz er kun cirka ét sekunds
observeret stabilitet, og en GoPro kan stadig fejle selv med inaktive
readiness-flags og en synlig mediepost.

## Hvad GoPro faktisk specificerer

- GoPros [State Management](https://gopro.github.io/OpenGoPro/docs/ble/protocol/state_management/)
  siger, at et kamera kan afvise kommandoer afhængigt af tilstand, og at bedste
  praksis er at vente på, at både System Busy og Encoding Active er nul, før man
  sender andet end status-/setting-forespørgsler.
- GoPros [statusspecifikation](https://gopro.github.io/OpenGoPro/docs/ble/statuses/)
  definerer blandt andet Busy (8), Encoding (10), Minimum Status Poll Period (60)
  og Ready (82). SDK-dokumentationen præciserer, at Minimum Status Poll Period er
  kameraets minimumsinterval for statusopdateringer og ikke bør underskrides
  ([Python SDK API reference](https://gopro.github.io/OpenGoPro/python_sdk/api.html)).
- BLE-advertisementet har desuden en Media Offload Status med blandt andet
  Available, New Media Available, SD Card OK og Busy
  ([BLE Setup](https://gopro.github.io/OpenGoPro/docs/ble/protocol/ble_setup/#advertisements)).
  Den er ikke et filspecifikt HTTP-readiness-signal og er ikke direkte anvendelig
  i BearVisions wired-only downloadsti.
- Det officielle [HTTP API](https://gopro.github.io/OpenGoPro/http/http.html)
  beskriver medieliste- og filendepunkterne, mens Python-API'et eksponerer
  `get_media_list`, `get_media_metadata` og `download_file`. De gennemgåede
  officielle specifikationer beskriver ikke et særskilt "media finalized"-flag,
  et stabilitetskrav eller en endpoint-specifik betydning af 503.
- GoPros [FAQ](https://gopro.github.io/OpenGoPro/docs/faq/) nævner eksplicit en
  fejl i størrelsen fra Media Info over 32-bit-grænsen og anbefaler filstørrelsen
  fra medielisten eller en `HEAD`-forespørgsel direkte mod filressourcen. Det gør
  `HEAD` til den bedste officielle byggesten til en readiness-probe. For korte
  BearVision-klip kan `get_media_metadata().file_size` være fallback, men det er
  ikke den generelt sikre størrelse. Medielistens rå `s`-felt indeholder den
  fulde filstørrelse, også for store MP4-filer, som vist i
  [OpenGoPro issue #887](https://github.com/gopro/OpenGoPro/issues/887#L168-L195).
- HTTP-standarden definerer 503 som, at serveren midlertidigt ikke kan håndtere
  forespørgslen, og at et eventuelt `Retry-After` skal angive ventetiden
  ([RFC 9110, section 15.6.4](https://www.rfc-editor.org/rfc/rfc9110.html#name-503-service-unavailable)).
  Det understøtter retry, men beviser ikke, hvorfor GoProen returnerede 503.

## Hvad konkrete implementationer gør

### GoPros officielle Python SDK og demo

Det officielle SDK's `WiredGoPro` venter før almindelige kommandoer på
`ENCODING=false` og `BUSY=false`; standardintervallet for den polling er to
sekunder
([kilde](https://github.com/gopro/OpenGoPro/blob/main/demos/python/sdk_wireless_camera_control/open_gopro/gopro_wired.py)).
Den officielle video-demo stopper optagelsen, henter straks medielisten, finder
set-differencen og downloader straks filen
([video.py](https://github.com/gopro/OpenGoPro/blob/main/demos/python/sdk_wireless_camera_control/open_gopro/demos/video.py)).

SDK'ets JSON-GET har retries for forbindelsesfejl, men den binære
downloadfunktion `_get_stream` laver ét streamende GET og kalder
`raise_for_status()` uden status-retry
([gopro_base.py](https://github.com/gopro/OpenGoPro/blob/main/demos/python/sdk_wireless_camera_control/open_gopro/gopro_base.py)).
En 503 fra fil-GET bliver derfor afleveret direkte som en exception. BearVision
er låst til `open-gopro` 0.22.0 i `uv.lock`; den installerede 0.22.0-kilde har
samme adfærd.

### GoPros issue tracker og community-kode

- [OpenGoPro issue #685](https://github.com/gopro/OpenGoPro/issues/685) viser
  gentagne USB-download-timeouts i den officielle video-demo. I det viste trace
  var filen allerede i medielisten, og kameraet rapporterede `BUSY=false`,
  `ENCODING=false` og `READY=true`. Fejlen opstod ofte efter mere end 90 % af
  filen var overført. Det er direkte evidens for, at status-gaten og medielisten
  ikke alene er en download-ready-kontrakt.
- [OpenGoPro issue #592](https://github.com/gopro/OpenGoPro/issues/592) beskriver
  intermitterende 404 fra medielisten selv efter kontrol af ready, not busy, not
  encoding og SD-card ready. I den fejltilstand krævede kameraet genstart. Det
  taler både for begrænset retry og for en tydelig terminal fejl efter deadline;
  polling må ikke fortsætte uendeligt.
- Et konkret script i [OpenGoPro discussion #265](https://github.com/gopro/OpenGoPro/discussions/265)
  indsætter en fast ventetid på to sekunder efter stop, fordi en hurtig
  medielisteforespørgsel ellers fejler. Det er en observeret workaround, ikke en
  specifikation.
- Den uofficielle [gopro-py-api](https://github.com/KonradIT/gopro-py-api/blob/master/goprocam/GoProCamera.py)
  venter efter shutter-stop på, at kameraets Busy-status bliver nul, og afviser
  download mens kameraet optager. Dens downloadsti har dog ikke
  filstørrelsesstabilitet eller 503-retry. Det er derfor ikke et mønster,
  BearVision bør kopiere ukritisk.

## Vurdering af 2 Hz og tre stabile målinger

`2 Hz` er acceptabelt for kamera-status, hvis Status 60 rapporterer højst 500 ms;
intervallet bør ellers sættes til `max(500 ms, minimum_status_poll_period)`.
Status 60 gælder statuspolling, ikke nødvendigvis `HEAD`, men samme begrænsning er
en fornuftig belastningsgrænse for den lille HTTP-server.

"Stabil" bør betyde følgende tuple fra den præcise filressource:

```text
(HTTP 200, Content-Length > 0, ETag hvis tilgængelig, Last-Modified hvis tilgængelig)
```

Brug hellere en tidsbaseret periode end kun et antal samples. Et praktisk
udgangspunkt er mindst 1,5-2,0 sekunders uændret længde med polling hver 500 ms.
Det er et engineering-valg, ikke et tal fra GoPro-specifikationen, og bør
kalibreres fra telemetry på det faktiske kamera, firmware og kliplængder.

Medielistens `mod`-felt bør ikke være det primære stabilitetssignal. Det har kun
sekundopløsning, og discussion #265 rapporterer betydelig drift i `cre`/`mod`.
`get_media_metadata().file_size` er bedre, men `HEAD Content-Length` er mere
direkte og undgår GoPros dokumenterede store-filproblem i Media Info.

## Anbefalet algoritme til BearVision

1. Kald `stop_recording()` og vent med en samlet deadline, eksempelvis 30
   sekunder.
2. Vent på `READY=true`, `BUSY=false` og `ENCODING=false`, uden at polle hurtigere
   end Status 60 tillader. SDK'et gør normalt allerede Busy/Encoding-kontrollen
   før den næste almindelige kommando; den samlede kontrol bør stadig være
   synlig i integrationskontrakten og testbar.
3. Poll medielisten, indtil netop én forventet ny MP4 er fundet. Fasthold den
   valgte sti i resten af operationen.
4. Poll `HEAD /videos/DCIM/{camera_file}`. Kræv om muligt, at `Content-Length`
   matcher medielistens rå `s`-felt. Nulstil stabilitetsvinduet ved ikke-200,
   manglende/ugyldig længde eller ændret længde/validator. Respektér
   `Retry-After`, hvis GoPro sender den.
5. Når vinduet er stabilt, stream til en unik `.part`-fil. Retry samme GET ved
   503, forbindelsesreset og read-timeout med jitteret backoff, eksempelvis
   0,5 s, 1 s, 2 s og 4 s, dog altid inden for den samlede deadline. Genstart
   ikke optagelsen, og genvælg ikke mediefilen.
6. Efter et tilsyneladende succesfuldt GET: sammenlign lokalt byteantal med GET's
   `Content-Length`, lav en ny `HEAD`, og kræv at længden stadig matcher. Kør
   derefter FFprobe/den eksisterende `MediaProbe`, så en trunkeret eller endnu
   ikke finaliseret MP4 ikke publiceres.
7. Omdøb `.part` atomisk til destinationen. Ved udløbet deadline returneres en
   særskilt transient camera/media-unavailable-fejl med antal forsøg, sidste
   HTTP-status og observerede størrelser.

Delvis genoptagelse med HTTP Range bør kun tilføjes, hvis kameraet faktisk
annoncerer/overholder Range og en stærk validator viser, at filen ikke har ændret
sig. Ellers er et nyt fuldt GET sikrere.

## Relevans for den nuværende kode

`src/bearvision/adapters/gopro.py` kalder i dag `stop_recording()`, henter én
medieliste og starter derefter én download. `src/bearvision/integrations/async_gopro.py`
videresender downloaden direkte til SDK'et. Der er derfor ingen BearVision-ejet
fil-readiness-probe, retry, `.part`-commit eller efterfølgende længdeverifikation
mellem linjerne.

Den bevarede Edge-kørselslog fra 4. september 2026 viser seks på hinanden
følgende 503-fejl for seks forskellige filer, `GX010615.MP4` til
`GX010620.MP4`, med cirka ti sekunders mellemrum. Det viser, at den eksisterende
komponent-retry gentog hele capture-operationen i stedet for at fastholde og
genprøve download af den første fil. Det er uønsket: en transient downloadfejl
må ikke skabe nye optagelser eller ændre capture-vinduet.

Samme log viser fortsatte `person_detected`-events mellem downloadforsøgene, så
preview-stien var aktiv, mens medieoverførslen fejlede. GoPros offentlige spec
angiver ikke klart, om fil-download skal fungere samtidig med den konkrete
wired preview-tilstand. Før en produktionsændring bør et isoleret hardwareforsøg
derfor sammenligne `HEAD`/download af den samme fil med preview henholdsvis
aktivt og stoppet. Hvis filen bliver tilgængelig straks efter `stop_preview`, er
preview-konflikten den primære årsag; hvis ikke, er bounded readiness-polling og
download-retry stadig nødvendig.

`open-gopro` 0.22.0's typed `MediaItem` eksponerer ikke det rå `s`-felt for en
almindelig video, og BearVisions `list_videos()` reducerer desuden posterne til
filnavne. Hvis `s` skal sammenlignes med `HEAD`, skal integrationen derfor bevare
den rå mediepost eller udvide modellen i integrationslaget.

Den mindste robuste ændring er at lægge readiness og retry i
`AsyncGoProController`, hvor HTTP/SDK-detaljerne hører hjemme, og lade
`GoProCameraAdapter` fortsat eje capture-forløbet og valget af den præcise nye
mediefil.
