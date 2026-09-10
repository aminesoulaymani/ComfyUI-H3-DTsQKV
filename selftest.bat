@echo off
rem Autotest du module H3-DTsQKV avec le Python embarque du portable (aucun poids lu, zero memoire GPU)
cd /d "%~dp0..\..\.."
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-H3-DTsQKV\selftest.py %*
pause
