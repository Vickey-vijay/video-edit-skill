@echo off
rem Double-click: edits every video dropped into the inbox folder, until this window is closed.
call "%~dp0vedit.cmd" watch %*
pause
