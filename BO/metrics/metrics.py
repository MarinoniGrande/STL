# metrics.py
# Collects system/process/GPU metrics and saves a JSON report.
# Safe to use even if some optional libs are missing.

import os, json, time, platform, socket, datetime, traceback
from contextlib import suppress

# Optional deps (auto-skip if missing)
with suppress(Exception):
    import psutil
with suppress(Exception):
    import GPUtil
with suppress(Exception):
    import pynvml
with suppress(Exception):
    import torch
with suppress(Exception):
    import tensorflow as tf
with suppress(Exception):
    import resource  # Unix only

# ---------- small utils ----------

def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default

def _now_iso():
    return datetime.datetime.now().astimezone().isoformat()

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

# ---------- collectors ----------

def _cpu_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": _safe(lambda: os.cpu_count()),
        "hostname": socket.gethostname(),
    }
    if hasattr(os, "getloadavg"):
        with suppress(Exception):
            la1, la5, la15 = os.getloadavg()
            info.update({"loadavg_1m": la1, "loadavg_5m": la5, "loadavg_15m": la15})
    if "psutil" in globals():
        with suppress(Exception):
            freq = psutil.cpu_freq()
            info["cpu_freq_mhz"] = freq._asdict() if freq else None
            info["cpu_percent"] = psutil.cpu_percent(interval=0.2)
    return info

def _ram_info():
    data = {}
    if "psutil" in globals():
        with suppress(Exception):
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            data["virtual_memory"] = vm._asdict()
            data["swap_memory"] = sm._asdict()
        with suppress(Exception):
            p = psutil.Process(os.getpid())
            mi = p.memory_info()
            data["process_memory"] = {
                "rss": mi.rss,
                "vms": mi.vms,
                "shared": getattr(mi, "shared", None),
                "data": getattr(mi, "data", None),
                "uss": None,
            }
            # Try USS
            with suppress(Exception):
                full = p.memory_full_info()
                data["process_memory"]["uss"] = getattr(full, "uss", None)
    return data

def _disk_info():
    data = {}
    if "psutil" in globals():
        with suppress(Exception):
            data["partitions"] = [p._asdict() for p in psutil.disk_partitions(all=False)]
        with suppress(Exception):
            data["usage_root"] = psutil.disk_usage("/")._asdict()
        with suppress(Exception):
            io = psutil.disk_io_counters(perdisk=False)
            data["disk_io_counters"] = io._asdict() if io else None
    return data

def _net_info():
    data = {}
    if "psutil" in globals():
        with suppress(Exception):
            nio = psutil.net_io_counters(pernic=False)
            data["net_io_counters"] = nio._asdict() if nio else None
    return data

def _process_info():
    data = {}
    if "psutil" in globals():
        with suppress(Exception):
            p = psutil.Process(os.getpid())
            data["pid"] = p.pid
            data["cmdline"] = p.cmdline()
            data["num_threads"] = p.num_threads()
            data["cpu_times"] = p.cpu_times()._asdict()
            with suppress(Exception):
                data["io_counters"] = p.io_counters()._asdict()
            with suppress(Exception):
                data["open_files_count"] = len(p.open_files())
    if "resource" in globals():
        with suppress(Exception):
            ru = resource.getrusage(resource.RUSAGE_SELF)
            data["resource_usage"] = {
                "utime_user": ru.ru_utime,
                "utime_system": ru.ru_stime,
                "max_rss_kb": getattr(ru, "ru_maxrss", None),
                "minor_faults": getattr(ru, "ru_minflt", None),
                "major_faults": getattr(ru, "ru_majflt", None),
                "inblock": getattr(ru, "ru_inblock", None),
                "oublock": getattr(ru, "ru_oublock", None),
                "nsignals": getattr(ru, "ru_nsignals", None),
                "nvcsw": getattr(ru, "ru_nvcsw", None),
                "nivcsw": getattr(ru, "ru_nivcsw", None),
            }
    return data

def _gpu_info():
    data = {"torch": {}, "tensorflow": {}, "nvidia": {}, "gputil": {}}

    if "torch" in globals():
        with suppress(Exception):
            data["torch"]["is_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                devs = []
                for i in range(torch.cuda.device_count()):
                    with suppress(Exception):
                        devs.append({
                            "index": i,
                            "name": torch.cuda.get_device_name(i),
                            "mem_allocated": torch.cuda.memory_allocated(i),
                            "mem_reserved": torch.cuda.memory_reserved(i),
                            "capability": torch.cuda.get_device_capability(i),
                        })
                data["torch"]["device_count"] = len(devs)
                data["torch"]["devices"] = devs
                data["torch"]["cuda_version"] = torch.version.cuda

    if "tf" in globals():
        with suppress(Exception):
            phys = tf.config.list_physical_devices('GPU')
            data["tensorflow"]["visible_gpus"] = [getattr(g, "name", str(g)) for g in phys]
            logical = tf.config.list_logical_devices('GPU')
            data["tensorflow"]["logical_gpus"] = [getattr(g, "name", str(g)) for g in logical]
            data["tensorflow"]["version"] = tf.__version__

    if "pynvml" in globals():
        try:
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            devs = []
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
                devs.append({
                    "index": i,
                    "name": _safe(lambda: pynvml.nvmlDeviceGetName(h).decode()),
                    "memory_total": mem.total,
                    "memory_used": mem.used,
                    "memory_free": mem.free,
                    "utilization_gpu": getattr(util, "gpu", None) if util else None,
                    "utilization_mem": getattr(util, "memory", None) if util else None,
                    "temperature": _safe(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
                    "power_usage_mw": _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(h)),
                })
            data["nvidia"]["devices"] = devs
            data["nvidia"]["driver_version"] = _safe(lambda: pynvml.nvmlSystemGetDriverVersion().decode())
            data["nvidia"]["nvml_version"] = _safe(lambda: pynvml.nvmlSystemGetNVMLVersion().decode())
        except Exception as e:
            data["nvidia"]["error"] = f"{type(e).__name__}: {e}"
        finally:
            with suppress(Exception):
                pynvml.nvmlShutdown()

    if "GPUtil" in globals():
        with suppress(Exception):
            gpus = GPUtil.getGPUs()
            data["gputil"]["devices"] = [{
                "id": g.id, "name": g.name, "load": g.load,
                "memory_total": g.memoryTotal, "memory_used": g.memoryUsed,
                "temperature": g.temperature, "uuid": g.uuid
            } for g in gpus]
    return data

def _pkg_info():
    info = {}
    if "torch" in globals():
        with suppress(Exception):
            info["torch"] = {
                "version": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn_enabled": getattr(torch.backends.cudnn, "enabled", None)
            }
    if "tf" in globals():
        with suppress(Exception):
            info["tensorflow"] = {"version": tf.__version__}
    return info

# ---------- public API ----------

def snapshot_system():
    return {
        "timestamp": _now_iso(),
        "cpu": _cpu_info(),
        "ram": _ram_info(),
        "disk": _disk_info(),
        "net": _net_info(),
        "process": _process_info(),
        "gpu": _gpu_info(),
        "packages": _pkg_info(),
    }

def delta_counters(start_snap, end_snap):
    out = {}
    try:
        s_io = (((start_snap or {}).get("disk") or {}).get("disk_io_counters") or {})
        e_io = (((end_snap or {}).get("disk") or {}).get("disk_io_counters") or {})
        out["disk_io_delta"] = {
            k: (e_io.get(k) - s_io.get(k)) if (k in e_io and k in s_io and isinstance(e_io.get(k), (int, float)) and isinstance(s_io.get(k), (int, float))) else None
            for k in ["read_count", "write_count", "read_bytes", "write_bytes", "read_time", "write_time", "busy_time"]
        }
    except Exception:
        pass
    try:
        s_net = (((start_snap or {}).get("net") or {}).get("net_io_counters") or {})
        e_net = (((end_snap or {}).get("net") or {}).get("net_io_counters") or {})
        out["net_io_delta"] = {
            k: (e_net.get(k) - s_net.get(k)) if (k in e_net and k in s_net and isinstance(e_net.get(k), (int, float)) and isinstance(s_net.get(k), (int, float))) else None
            for k in ["bytes_sent", "bytes_recv", "packets_sent", "packets_recv", "errin", "errout", "dropin", "dropout"]
        }
    except Exception:
        pass
    try:
        s_proc = (((start_snap or {}).get("process") or {}).get("io_counters") or {})
        e_proc = (((end_snap or {}).get("process") or {}).get("io_counters") or {})
        if s_proc and e_proc:
            out["process_io_delta"] = {
                k: (e_proc.get(k) - s_proc.get(k)) if (k in e_proc and k in s_proc) else None
                for k in ["read_count", "write_count", "read_bytes", "write_bytes"]
            }
    except Exception:
        pass
    return out

class PerfTimer:
    """Context manager OR manual .enter()/.exit() for job metrics."""
    def __enter__(self):
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()
        self.start_snap = snapshot_system()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.end_wall = time.perf_counter()
        self.end_cpu = time.process_time()
        self.end_snap = snapshot_system()
        self.wall_time_s = self.end_wall - self.start_wall
        self.cpu_time_s = self.end_cpu - self.start_cpu
        self.deltas = delta_counters(self.start_snap, self.end_snap)

def build_report(timer: PerfTimer, notes: str | None = None, error: str | None = None):
    return {
        "timestamp_end": _now_iso(),
        "wall_time_seconds": getattr(timer, "wall_time_s", None),
        "cpu_time_seconds": getattr(timer, "cpu_time_s", None),
        "start_snapshot": getattr(timer, "start_snap", None),
        "end_snapshot": getattr(timer, "end_snap", None),
        "delta_counters": getattr(timer, "deltas", None),
        "notes": notes or "",
        "error": error,  # full traceback if you pass it
    }

def save_metrics_report(out_dir: str, report: dict, filename: str = "METRICAS.json"):
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path
