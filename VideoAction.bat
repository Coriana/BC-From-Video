for /F "delims=" %%x in (VideoFiles.txt) do (
     python Video_action.py --video-path="%%x"
)

pause