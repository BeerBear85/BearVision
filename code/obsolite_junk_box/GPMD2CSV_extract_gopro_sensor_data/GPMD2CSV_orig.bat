if [%1]==[] goto :eof
:loop
START /WAIT .\ffmpeg -y -i "%~1" -codec copy -map 0:m:handler_name:"	GoPro MET" -f rawvideo "%~n1".bin
START /WAIT .\gpmdinfo -i "%~n1".bin
DEL "%~n1".bin
shift
if not [%1]==[] goto loop