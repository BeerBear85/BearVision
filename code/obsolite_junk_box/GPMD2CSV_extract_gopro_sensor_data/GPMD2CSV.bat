if [%1]==[] goto :eof
:loop
START /WAIT .\ffmpeg -y -i "%~1" -codec copy -map 0:m:handler_name:"	GoPro MET" -f rawvideo "%~n1".bin
::START /WAIT .\gpmdinfo -i "%~n1".bin
START /WAIT .\gopro2json -i "%~n1".bin -o "%~n1".json
::DEL "%~n1".bin
shift
if not [%1]==[] goto loop