@echo off
rem 控制台切换到 GBK, 保证中文正常显示 / switch console to GBK for Chinese text
chcp 936 >nul
setlocal
title 手机录屏内容提取 - Web服务启动器
rem 切换到脚本所在目录(即项目根目录) / cd to script dir (project root)
cd /d "%~dp0"

echo.
echo ================================================================
echo   手机录屏内容提取 - 前后端一键启动脚本
echo   项目根目录: %CD%
echo ================================================================
echo.

rem ================= 1. 定位 Python 解释器 =================
rem 本项目必须使用 conda 环境 gnt: 它是唯一同时安装 av(音频解码)
rem 与 fastapi/uvicorn(Web服务) 的环境。系统 python(base)缺少 av,
rem 一旦被选中会导致服务启动即崩溃(ModuleNotFoundError: No module named 'av')。
rem This project requires conda env "gnt": it is the only env that has
rem both av (audio decode) and fastapi/uvicorn (web service).
rem 注意: conda 是 conda.bat, 在批处理中必须用 call 或在子进程(for /f)里调用,
rem 否则控制权会被 conda.bat 接管, 导致本脚本提前退出、窗口一闪而过。
set "PYTHON="

rem 1a) 首选 conda 环境 gnt: 用 for /f 子进程解析解释器绝对路径
rem     Prefer conda env "gnt"; resolve interpreter path in a sub-shell.
for /f "usebackq delims=" %%p in (`conda run -n gnt python -c "import sys;print(sys.executable)" 2^>nul`) do set "PYTHON=%%p"
if not defined PYTHON goto PY_FALLBACK

"%PYTHON%" -c "import av, fastapi, uvicorn, multipart, websockets" >nul 2>&1
if not errorlevel 1 goto PY_OK

:PY_FALLBACK
rem 1b) 回退: 使用 PATH 中的 python, 但同样必须同时具备 av 与 Web 依赖
rem     Fallback: python on PATH; must also provide av + web deps.
set "PYTHON=python"
python -c "import av, fastapi, uvicorn, multipart, websockets" >nul 2>&1
if not errorlevel 1 goto PY_OK

echo [x] 未找到可用的 Python 环境
echo     需要同时安装: av / fastapi / uvicorn / python-multipart / websockets
echo     推荐安装到 conda 环境 gnt:
echo     conda run -n gnt python -m pip install -r web\backend\requirements.txt
pause
exit /b 1

:PY_OK
echo [ok] Python 解释器: %PYTHON%

rem ================= 2. 前端构建检查 =================
rem 有 dist 就跳过(未改动前端时), 自动 npm install + build; 无 dist 则构建。
rem Rebuild only when dist is missing; skip otherwise for fast startup.
set "DIST=web\frontend\dist\index.html"
if exist "%DIST%" (
    echo [ok] 前端已构建, 跳过构建
    echo     修改前端源码后, 请删除 web\frontend\dist 或重新运行本脚本以重建。
) else (
    echo [i] 前端尚未构建, 开始构建...
    pushd web\frontend
    if not exist node_modules (
        echo [i] 安装前端依赖 npm install ...
        call npm install
    )
    call npm run build
    popd
    if not exist "%DIST%" (
        echo [x] 前端构建失败, 请确认 node/npm 已安装
        pause
        exit /b 1
    )
)

rem ================= 3. 启动后端服务(独立窗口) =================
rem 后端放在独立控制台窗口运行, 该窗口由 cmd /k 持有: 服务退出后窗口仍会保留,
rem 日志与报错不会被吞掉; 启动器窗口即使立刻关闭也不影响服务运行。
rem The backend runs in its own console window held by cmd /k, so logs and
rem errors stay visible even after the launcher window is gone.
rem --host 0.0.0.0 = 监听所有网卡, 手机可访问本机IP地址 http://192.168.x.x:8000
rem 不自动打开浏览器, 请手动用 http:// 访问 / no auto browser open.
echo.
echo [i] 正在独立窗口启动后端服务...
echo.
echo     ============================================================
echo      在浏览器打开, 务必用 http, 不要用 https:
echo        本机:    http://127.0.0.1:8000
echo        本机:    http://localhost:8000
echo        局域网:  http://本机IP:8000     同一局域网内的手机等设备
echo      停止服务:  关闭 "手机录屏内容提取 - Web服务" 窗口,
echo                 或双击 "web服务一键停止脚本.bat"
echo     ============================================================
echo.
start "手机录屏内容提取 - Web服务" cmd /k ""%PYTHON%" web\run.py --host 0.0.0.0"

echo [ok] 服务已在独立窗口启动, 本窗口可以关闭。
echo      服务日志与报错都在 "手机录屏内容提取 - Web服务" 窗口中。
pause
