"""
录屏内容提取 Web 服务便捷启动脚本 / Convenience launcher.

用法 / Usage:
  python web/run.py [--port 8000] [--host 127.0.0.1]
"""

import argparse
import socket
import sys
from copy import deepcopy
from pathlib import Path

# 项目根目录加入路径(确保 import src.* 与 web.backend.* 均可用)
# Add project root to sys.path so both src.* and web.backend.* import.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn
from uvicorn.config import LOGGING_CONFIG

# 通配监听地址: 这类地址需要同时覆盖 IPv4 与 IPv6(见 _make_dualstack_socket)
# Wildcard bind hosts: these need to cover both IPv4 and IPv6.
_WILDCARD_HOSTS = ("0.0.0.0", "::", "")


def _plain_log_config() -> dict:
    """复制 uvicorn 默认日志配置并关闭 ANSI 颜色 / uvicorn log config, no color.

    Windows 的 cmd 控制台默认不解析 ANSI 转义序列, uvicorn 的彩色日志会把
    "[32mINFO[0m" 原样打印出来(其中的 ESC 字节还会显示成方框)。这里显式关闭
    颜色, 保证一键启动脚本窗口里的日志干净可读。
    Windows cmd does not interpret ANSI escapes by default, so uvicorn's
    colorized logs appear verbatim as "[32mINFO[0m" (ESC bytes as boxes).
    Disable colors so the launcher window logs stay readable.
    """
    cfg = deepcopy(LOGGING_CONFIG)
    for formatter in cfg.get("formatters", {}).values():
        formatter["use_colors"] = False
    return cfg


def _is_port_in_use(port: int) -> bool:
    """探测本机端口是否已被监听 / whether the port is already listening.

    Windows 下 SO_REUSEADDR 允许绑定"已被占用"的端口, 若不做预检, 重复启动会
    悄悄起第二个服务(两个进程抢同一端口), 表现为"启动了但行为诡异"。
    这里先用 TCP 探测一次, 让端口冲突直接以明确提示暴露出来。
    On Windows SO_REUSEADDR lets a second bind silently overlap an existing
    listener, so probe the port first and fail loudly on conflict.

    注意: 必须用 create_connection 而不是 socket.connect_ex —— 设置了超时的套接字
    处于非阻塞模式, connect_ex 会立刻返回 WSAEWOULDBLOCK(10035) 而非 0。
    Use create_connection (not connect_ex on a timeout socket): a non-blocking
    connect_ex returns WSAEWOULDBLOCK instead of 0 even when the port is open.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _make_dualstack_socket(port: int) -> socket.socket:
    """创建 IPv4/IPv6 双栈监听套接字 / Create a dual-stack listening socket.

    背景 / Why:
      默认按 IPv4 绑定(0.0.0.0)时, 浏览器访问 http://localhost:8000 会先解析到
      IPv6 的 ::1; 由于服务端只在 IPv4 上监听, 连接被拒绝, 于是出现
      "127.0.0.1 能开、localhost 打不开"的现象。
      Windows 上 IPV6_V6ONLY 默认为 1, 直接绑 :: 又会丢掉 IPv4。
      因此显式关闭 IPV6_V6ONLY, 用一个 :: 套接字同时接受 IPv4 与 IPv6
      (IPv4 走 IPv4-mapped 地址), 使 127.0.0.1 / localhost / 局域网IP 全部可用。
    """
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 关闭 v6only: 让同一个套接字同时服务 IPv4 与 IPv6
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.bind(("::", port))
    sock.listen(socket.SOMAXCONN)
    return sock


def main() -> None:
    parser = argparse.ArgumentParser(description="启动录屏提取Web服务 / Run the web demo")
    parser.add_argument("--host", default="0.0.0.0",
                        help="监听地址: 0.0.0.0=局域网可访问(演示用) / "
                             "127.0.0.1=仅本机 / bind host")
    parser.add_argument("--port", type=int, default=8000,
                        help="监听端口 / bind port")
    parser.add_argument("--reload", action="store_true",
                        help="开发热重载 / hot reload (dev)")
    args = parser.parse_args()

    print("=" * 56)
    print("手机录屏内容提取 - 演示服务 / Screen-Recorder Extraction Demo")
    print(f"访问 / open: http://{args.host}:{args.port}")
    print("=" * 56)

    # 端口冲突预检: 避免 SO_REUSEADDR 下静默起第二个服务
    # Pre-flight port check: never silently start a second server.
    if _is_port_in_use(args.port):
        print(f"[x] 端口 {args.port} 已被占用, 未启动新服务。")
        print("     请先关闭占用该端口的程序(通常是上一次未关闭的服务窗口), 再重试。")
        sys.exit(2)

    config = uvicorn.Config("web.backend.main:app",
                            host=args.host,
                            port=args.port,
                            reload=args.reload,
                            workers=1,
                            log_config=_plain_log_config())

    # 通配地址下优先用双栈套接字, 保证 localhost(::1) 与 127.0.0.1 均可访问;
    # reload 模式由 uvicorn 自己管理子进程套接字, 故不接管。
    # Prefer a dual-stack socket for wildcard hosts so that both localhost
    # (::1) and 127.0.0.1 work; skip in reload mode (uvicorn owns sockets).
    if args.host in _WILDCARD_HOSTS and not args.reload:
        try:
            sock = _make_dualstack_socket(args.port)
        except OSError as exc:
            print(f"[i] 双栈监听不可用, 回退为 {args.host} 单栈: {exc}")
        else:
            uvicorn.Server(config).run(sockets=[sock])
            return

    # 非双栈路径: reload 模式交给 uvicorn 自管理, 其余用已构建的 config 直接跑。
    # Non dual-stack path: reload is delegated to uvicorn, otherwise run config.
    if args.reload:
        uvicorn.run("web.backend.main:app",
                    host=args.host,
                    port=args.port,
                    reload=True,
                    workers=1,
                    log_config=_plain_log_config())
        return

    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
