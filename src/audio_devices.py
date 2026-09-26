"""
audio_devices.py -- input-device listing/selection shared by edge_agent.py and
record_probe.py.

Pinning the mic by index (rather than following the Windows default) keeps a
demo from silently switching to a headset that gets plugged in, and keeps
recordings and live inference on the same capture chain -- the wake-word model
is sensitive to the chain's spectral signature (see sanity_check.py's
"live chain" realism condition).
"""

from __future__ import annotations

import sounddevice as sd


def list_input_devices() -> None:
    """Print every input-capable device: index, name, host API, default rate."""
    hostapis = sd.query_hostapis()
    default_in = sd.default.device[0]
    print(f"{'idx':>4}  {'host API':22s} {'rate':>7}  name")
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue
        marker = "  <- Windows default" if index == default_in else ""
        print(f"{index:>4}  {hostapis[dev['hostapi']]['name']:22s} "
              f"{int(dev['default_samplerate']):>7}  {dev['name']}{marker}")


def resolve_input_device(device: int | None, sample_rate: int) -> tuple[int, str]:
    """Validate that `device` (None = Windows default) can capture mono
    float32 at `sample_rate`, and return (index, "name [host API]").

    Fails at startup with a readable message instead of mid-demo -- WASAPI
    and WDM-KS devices often refuse a rate other than their native one,
    where MME/DirectSound resample transparently."""
    index = sd.default.device[0] if device is None else device
    try:
        dev = sd.query_devices(index, kind="input")
    except (ValueError, sd.PortAudioError) as exc:
        raise SystemExit(f"input device {index!r} not usable: {exc}\n"
                         f"run with --list-devices to see valid indices")
    try:
        sd.check_input_settings(device=index, samplerate=sample_rate,
                                channels=1, dtype="float32")
    except sd.PortAudioError as exc:
        raise SystemExit(f"input device {index} ({dev['name']}) rejects "
                         f"{sample_rate} Hz mono float32: {exc}\n"
                         f"try the same mic under its MME or DirectSound index")
    hostapi = sd.query_hostapis(dev["hostapi"])["name"]
    return index, f"{dev['name']} [{hostapi}]"
