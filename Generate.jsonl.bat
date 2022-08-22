for /F "delims=" %%x in (VideoFiles.txt) do (
     python IDM_Video.py --video-path="%%x"
)

pause