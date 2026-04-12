# Code Signing Policy

This document describes the code signing policy for **Pokemon Champions OBS Overlay**.

## Project Information

- **Project Name**: Pokemon Champions OBS Overlay
- **Repository**: https://github.com/minuspiral/pokemon-champions-obs-overlay
- **License**: MIT
- **Maintainer**: minuspiral ([GitHub](https://github.com/minuspiral))

## Signed Artifacts

The following binary artifacts are signed:

- `PokemonOverlay.exe` — Windows GUI executable built from `overlay.py`
- Distributed in two forms:
  - `PokemonOverlay-nuitka.zip` — Nuitka native-compiled standalone distribution (recommended)
  - `PokemonOverlay.exe` — Single-file executable (PyInstaller)

All signed binaries are built from the source code in this repository and published via GitHub Releases.

## Build Process

1. **Source of Truth**: All code lives in the `main` branch of the GitHub repository.
2. **Build Reproducibility**: Builds are performed locally on the maintainer's Windows 11 machine using:
   - Python 3.10
   - Nuitka 4.0.8 with MSVC 14.5 backend (primary build)
   - PyInstaller (legacy single-file build)
   - Dependencies pinned in `requirements.txt`
3. **Build Commands**: Documented in `build.bat`.
4. **Artifacts**: Output is uploaded to GitHub Releases under versioned tags (e.g., `v1.3.0`).

## Signing Authorization

Only the repository maintainer (`minuspiral`) is authorized to:

- Build and publish releases
- Submit binaries for signing via SignPath Foundation
- Upload signed artifacts to GitHub Releases

## Security Measures

- **GitHub Account**: Protected with **multi-factor authentication (2FA)**
- **Repository Access**: Single maintainer, no external collaborators with write access
- **Signing Scope**: Only binaries built from this repository's source code are signed
- **No Code from Untrusted Sources**: All dependencies come from PyPI official packages or the Python standard library

## Incident Response

If unauthorized signed binaries are discovered, or a security incident occurs:

1. **Immediate Action**: Revoke the compromised signing credentials via SignPath support
2. **Notification**: Open a GitHub Issue tagged `security` in this repository
3. **Contact**: `oss-support@signpath.org` (SignPath Foundation)
4. **User Notification**: Publish a security advisory via GitHub Security Advisories

## Dependencies

All dependencies used for building are third-party open-source packages installed from PyPI:

- `opencv-python` — BSD License
- `numpy` — BSD License
- `Pillow` — HPND License
- `obsws-python` — GPL-3.0 License (used via its Python API, overlay code itself is MIT)

See `requirements.txt` for the full list.

## Version History

| Version | Release Date | Signed |
|---|---|---|
| v1.0.0 | 2026-04-11 | No |
| v1.0.1 | 2026-04-11 | No |
| v1.2.0 | 2026-04-12 | No |
| v1.3.0 | 2026-04-12 | Pending SignPath approval |

---

*This policy follows [SignPath Foundation guidelines for OSS projects](https://signpath.org/terms).*
