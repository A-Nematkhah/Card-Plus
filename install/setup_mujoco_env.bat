@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
set MUJOCO_PY_MUJOCO_PATH=C:\Users\AmirHossein\.mujoco\mujoco210
set MUJOCO_PY_MJKEY_PATH=C:\Users\AmirHossein\.mujoco\mjkey.txt
set PATH=C:\Users\AmirHossein\.mujoco\mujoco210\bin;%PATH%
if exist "D:\Thesis\Implementation\Card\xai_keys.json" (
  if not defined XAI_KEYS_PATH set XAI_KEYS_PATH=D:\Thesis\Implementation\Card\xai_keys.json
)
if not defined XAI_PROXY set XAI_PROXY=http://127.0.0.1:10809
"C:\Users\AmirHossein\miniconda3\envs\card\python.exe" %*
