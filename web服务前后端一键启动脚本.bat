@echo off
title 手机录屏内容提取 - Web服务一键启动
rem 切换到脚本所在目录(即项目根目录) / cd to script dir (project root)
cd /d "%~dp0"

echo.
echo ================================================================
echo   手机录屏内容提取 - 前后端一键启动脚本
echo   项目根目录: %CD%
echo ================================================================
echo.

rem ================= 1. 检测 Python 环境 =================
rem 优先用系统 python; 若缺依赖则回退到 conda 环境 gnt。
rem Prefer system python; fall back to conda env "gnt" if deps missing.
rem 注意: websockets 是 uvicorn 处理 WebSocket 升级所必需。
set "PYTHON=python"
python -c "import fastapi, uvicorn, multipart, websockets" >nul 2>&1
if "%errorlevel%"=="0" goto PY_OK

echo [i] 系统 python 缺少后端依赖, 尝试 conda 环境 gnt ...
conda run -n gnt python -c "import fastapi, uvicorn, multipart, websockets" >nul 2>&1
if "%errorlevel%"=="0" (
    set "PYTHON=conda run -n gnt python"
    goto PY_OK
)

echo [x] 未找到可用的 Python 环境。
echo     需要 fastapi / uvicorn / python-multipart / websockets。
echo     请先安装依赖: python -m pip install -r web\backend\requirements.txt
pause
exit /b 1

:PY_OK
echo [ok] Python 环境: %PYTHON%

rem ================= 2. 前端构建检查 =================
rem 若 dist 不存在(未构建过), 自动 npm install + build; 已构建则跳过以加速。
rem Rebuild only when dist is missing; skip otherwise for fast startup.
set "DIST=web\frontend\dist\index.html"
if exist "%DIST%" (
    echo [ok] 前端已构建, 跳过构建。
    echo     若修改了前端源码, 请删除 web\frontend\dist 后重新运行本脚本以重建。
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
        echo [x] 前端构建失败, 请检查 node/npm 环境。
        pause
        exit /b 1
    )
)

rem ================= 3. 启动后端并自动打开浏览器 =================
rem --host 0.0.0.0 = 绑定所有网卡, 局域网内可用本机IP访问(如 192.168.x.x:8000)。
echo.
echo [i] 启动后端服务(局域网可访问): http://0.0.0.0:8000
echo     本机访问:   http://127.0.0.1:8000
echo     局域网访问: http://本机IP:8000   (同网段其他设备)
echo     关闭本窗口即停止服务, 行为日志实时输出在下方。
echo.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8000'"
%PYTHON% web\run.py --host 0.0.0.0

echo.
echo [i] 服务已停止。
pause
