@echo off
rem ================================================================
rem  停止 Web 服务 / Stop the web service
rem ----------------------------------------------------------------
rem  启动脚本会把后端放在独立窗口("手机录屏内容提取 - Web服务")里运行;
rem  关闭"启动器"窗口并不会停止服务。本脚本按端口找到后端进程并结束它,
rem  然后关闭那个独立服务窗口, 从而真正停止服务。
rem  The launcher runs the backend in a separate console window; closing
rem  the launcher does NOT stop the service. This script finds the process
rem  listening on the port, kills it, then closes that window.
rem ================================================================

rem 控制台切换到 GBK, 保证中文正常显示 / switch console to GBK for Chinese text
chcp 936 >nul
setlocal
title 手机录屏内容提取 - Web服务停止器

set "PORT=8000"

echo.
echo ================================================================
echo   手机录屏内容提取 - 停止 Web 服务
echo ================================================================
echo.

rem ---- 1) 按端口定位并结束后端进程 ----
rem Find and kill the backend process listening on %PORT%.
set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /c:":%PORT% " ^| findstr /c:"LISTENING"') do (
    set "FOUND=1"
    echo [i] 停止服务进程 PID: %%p
    taskkill /f /t /pid %%p >nul 2>&1
)
if not defined FOUND echo [i] 端口 %PORT% 上没有正在运行的服务, 无需停止。

rem ---- 2) 关闭残留的独立服务窗口 ----
rem 关闭 "手机录屏内容提取 - Web服务" 窗口(若仍在) / close the backend window.
taskkill /f /fi "WINDOWTITLE eq 手机录屏内容提取 - Web服务" >nul 2>&1

echo.
echo [ok] 已尝试停止服务, 端口 %PORT% 现已释放。
echo      可刷新 http://127.0.0.1:%PORT% 确认已无法访问。
echo.
pause
