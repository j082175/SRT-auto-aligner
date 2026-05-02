@echo off
rem SRT 자동 정렬기 GUI 런처 (콘솔 창 없이 실행)
rem 디버그용으로 콘솔 보고 싶으면 pythonw.exe → python.exe 로 바꾸면 됨

cd /d "%~dp0"
start "" pythonw.exe main.py
