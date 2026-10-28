"""Self-check: streaming command listener delivers rapid successive writes.

Run from pi-control/ :  python check_stream_cmd.py
No Firebase needed — fake ref + fake stream callback.
"""
import asyncio
import os
import sys
import time
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec


# Fabricate any non-project module that isn't installed (serial, pigpio, ...)
# so the bridge can be imported and exercised on a dev box without the hardware
# stack. Project modules under `src` still must be real.
class _StubLoader(Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__getattr__ = lambda attr: type(attr, (), {})
        return mod

    def exec_module(self, module):
        pass


class _StubFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("src", "asyncio", "os", "sys", "time", "types"):
            return None
        return ModuleSpec(fullname, _StubLoader(), is_package=True)


sys.meta_path.append(_StubFinder())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.comms.firebase_bridge import FirebaseBridge, RemoteCommand  # noqa: E402


class FakeEvent:
    def __init__(self, data):
        self.data = data
        self.path = "/"


class FakeCommandsRef:
    def __init__(self):
        self.cb = None
        self.writes = []

    def listen(self, callback, error_callback=None):
        self.cb = callback
        return self

    def update(self, payload):
        self.writes.append(payload)

    def close(self):
        pass


class FakeRootRef:
    def __init__(self):
        self.commands = FakeCommandsRef()

    def child(self, name):
        assert name == "commands"
        return self.commands


async def main():
    bridge = FirebaseBridge(enabled=False)          # skips real init
    bridge._enabled = True
    bridge._ref = FakeRootRef()
    bridge._last_command_ts = 0.0
    bridge._cmd_loop = asyncio.get_running_loop()
    bridge._cmd_queue = asyncio.Queue()

    seen = []

    async def handler(cmd: RemoteCommand):
        seen.append((cmd.manual_surge, cmd.manual_sway))

    bridge.on_command(handler)
    assert bridge._start_listen() is True, "listen() not established"
    assert bridge._ref.commands.cb is not None, "callback not registered"

    # Two writes ~5 ms apart: joystick vector, then the release-to-zero write.
    # The old 500 ms poller would only ever see the second one.
    base = time.time()
    bridge._on_command_event(FakeEvent({"mode": "manual", "manual": {"surge": 0.8, "sway": -0.4, "yaw": 0},
                                        "timestamp": base}))
    bridge._on_command_event(FakeEvent({"mode": "manual", "manual": {"surge": 0, "sway": 0, "yaw": 0},
                                        "timestamp": base + 0.005}))
    # Replay of the same write must be dropped (also filters our own ack writes)
    bridge._on_command_event(FakeEvent({"mode": "manual", "manual": {"surge": 0, "sway": 0, "yaw": 0},
                                        "timestamp": base + 0.005}))

    await asyncio.sleep(0)  # let call_soon_threadsafe callbacks run
    while not bridge._cmd_queue.empty():
        await bridge._dispatch_command(await bridge._cmd_queue.get())

    assert seen == [(0.8, -0.4), (0.0, 0.0)], f"dropped writes: {seen}"
    assert [w.get("ack_mode") for w in bridge._ref.commands.writes] == ["manual", "manual"]

    # Non-dict event (node deleted) must be ignored, not raise
    bridge._on_command_event(FakeEvent(None))

    print("OK streaming dispatch:", seen)


asyncio.run(main())
