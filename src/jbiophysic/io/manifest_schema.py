"""Minimal manifest-schema constants for evidence artifacts."""

VALID_CURRENT_DENSITY_LAYOUTS = ("channel_first", "channel_last")


def validate_manifest_schema(manifest: dict) -> dict[str, object]:
    layout = manifest.get("array_layout", {}).get("current_density")
    failures = []
    if layout not in VALID_CURRENT_DENSITY_LAYOUTS:
        failures.append("array_layout.current_density must be channel_first or channel_last")
    return {"accepted": not failures, "failures": failures}
