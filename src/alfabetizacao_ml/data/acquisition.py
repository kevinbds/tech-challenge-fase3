"""Content-addressed downloads; replay is governed by hashes, never freshness."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

SOURCES = [
    {
        "id": f"aeeb_{year}",
        "url": f"https://download.inep.gov.br/dados_abertos/{name}",
        "filename": name,
        "observation_year": year,
        "kind": "aeeb_zip",
    }
    for year, name in [
        (2023, "microdados_avaliacao_da_alfabetizacao_2023.zip"),
        (2024, "microdados_avaliacao_da_alfabetizacao_2024.zip"),
        (2025, "microdados_AEEB_2025.zip"),
    ]
] + [
    {
        "id": "metas_uf_brasil_2024",
        "url": "https://download.inep.gov.br/alfabetiza_brasil/resultados_e_metas_ufs_2024_2.xlsx",
        "filename": "resultados_e_metas_ufs_2024_2.xlsx",
        "observation_year": 2024,
        "kind": "metas_uf_xlsx",
    },
    {
        "id": "metas_municipios_2024",
        "url": "https://download.inep.gov.br/alfabetiza_brasil/resultados_e_metas_municipios_2024.xlsx",
        "filename": "resultados_e_metas_municipios_2024.xlsx",
        "observation_year": 2024,
        "kind": "metas_municipio_xlsx",
    },
    {
        "id": "inse_2021",
        "url": "https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/2021/nivel_socioeconomico/INSE_2021_municipios.xlsx",
        "filename": "INSE_2021_municipios.xlsx",
        "observation_year": 2021,
        "kind": "inse_xlsx",
    },
    {
        "id": "ibge_censo_2022",
        "url": "https://servicodados.ibge.gov.br/api/v3/agregados/4714/periodos/2022/variaveis/93%7C614?localidades=N6[all]",
        "filename": "ibge_4714_2022.json",
        "observation_year": 2022,
        "kind": "ibge_censo_json",
    },
    {
        "id": "ibge_municipios",
        "url": "https://servicodados.ibge.gov.br/api/v1/localidades/municipios?orderBy=id",
        "filename": "ibge_municipios.json",
        "observation_year": None,
        "kind": "ibge_directory_json",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _download(url: str, destination: Path) -> dict:
    """Use curl for the official INEP host's TLS compatibility on Windows.

    HTTPS is always tried first. HTTP fallback is restricted to the exact public
    INEP download host and recorded; it does not imply TLS-authenticated content.
    Reproduction additionally requires the previously frozen content digest.
    """
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for official-source downloads")
    urls = [url]
    if urlparse(url).hostname == "download.inep.gov.br":
        urls.append(url.replace("https://", "http://", 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for candidate in urls:
        result = subprocess.run(
            [curl, "--globoff", "--fail", "--location", "--silent", "--show-error",
             "--connect-timeout", "20", "--max-time", "600", "--retry", "2",
             "--output", str(destination), candidate],
            capture_output=True, text=True, errors="replace", check=False,
        )
        if result.returncode == 0 and destination.is_file() and destination.stat().st_size:
            return {"download_url": candidate, "transport": urlparse(candidate).scheme}
        errors.append(f"{urlparse(candidate).scheme}: curl exit {result.returncode}")
    raise RuntimeError(f"Official source download failed ({'; '.join(errors)})")


def _validate_manifest_metadata(manifest: dict) -> None:
    if manifest.get("schema_version") != 1 or not manifest.get("sources"):
        raise ValueError("Unsupported or empty source manifest")
    sources = {source["id"]: source for source in SOURCES}
    entries = manifest["sources"]
    if len(entries) != len(sources) or {entry.get("id") for entry in entries} != set(sources):
        raise ValueError("Manifest source IDs are missing, duplicated or unexpected")
    for entry in entries:
        if any(entry.get(key) != value for key, value in sources[entry["id"]].items()):
            raise ValueError("Manifest source metadata differs from the official allowlist")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise ValueError("Invalid source digest")
        if type(entry.get("bytes")) is not int or entry["bytes"] <= 0:
            raise ValueError("Invalid source length")
        expected_path = f"bronze/{entry['id']}/{entry['sha256']}/{entry['filename']}"
        if entry.get("relative_path") != expected_path:
            raise ValueError("Source path escapes or differs from its content address")
    expected = canonical_hash([
        {key: entry[key] for key in ("id", "sha256", "bytes")} for entry in entries
    ])
    if manifest.get("content_id") != expected:
        raise ValueError("Manifest content_id mismatch")
    if manifest.get("snapshot_id") != f"local-{expected[:16]}":
        raise ValueError("Manifest snapshot_id mismatch")


def verify_manifest(cache_dir: Path, manifest_path: Path) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _validate_manifest_metadata(manifest)
    root = Path(cache_dir).resolve()
    for entry in manifest["sources"]:
        path = (root / entry["relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source path escapes the private cache")
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Frozen source integrity mismatch: {entry['id']}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Frozen source length mismatch: {entry['id']}")
    expected = canonical_hash([
        {key: entry[key] for key in ("id", "sha256", "bytes")}
        for entry in manifest["sources"]
    ])
    if manifest.get("content_id") != expected:
        raise ValueError("Manifest content_id mismatch")
    return manifest


def acquire_sources(
    cache_dir: Path,
    mode: str = "update",
    expected_manifest: Path | None = None,
) -> Path:
    """Acquire official inputs or replay a frozen manifest.

    An intact replay never recontacts the remote server or checks source age.
    Missing cached blobs are reacquired from the frozen URL and checked against
    the expected hash. Existing corrupted blobs cause an explicit failure.
    """
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if mode not in {"update", "reproduce"}:
        raise ValueError("mode must be update or reproduce")
    if mode == "reproduce":
        if expected_manifest is None:
            raise ValueError("Reproduction requires an expected manifest")
        manifest = json.loads(Path(expected_manifest).read_text(encoding="utf-8"))
        _validate_manifest_metadata(manifest)
        for entry in manifest["sources"]:
            path = (root / entry["relative_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Source path escapes the private cache")
            if not path.exists():
                temporary = root / "downloads" / f"{entry['id']}.part"
                _download(entry["url"], temporary)
                if sha256_file(temporary) != entry["sha256"]:
                    raise ValueError(f"Remote source changed: {entry['id']}")
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temporary), str(path))
        verify_manifest(root, Path(expected_manifest))
        return Path(expected_manifest)

    entries = []
    for source in SOURCES:
        temporary = root / "downloads" / f"{source['id']}.part"
        transport = _download(source["url"], temporary)
        digest = sha256_file(temporary)
        relative = Path("bronze") / source["id"] / digest / source["filename"]
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError("Content-addressed Bronze object was modified")
            temporary.unlink()
        else:
            shutil.move(str(temporary), str(destination))
        entries.append({
            **source, **transport, "sha256": digest,
            "bytes": destination.stat().st_size,
            "relative_path": relative.as_posix(),
            "acquired_at": datetime.now(UTC).isoformat(),
            "publication_date": None,
        })
        print(f"acquired {source['id']}: {destination.stat().st_size} bytes", flush=True)
    content_id = canonical_hash([
        {key: entry[key] for key in ("id", "sha256", "bytes")} for entry in entries
    ])
    manifest = {"schema_version": 1, "content_id": content_id,
                "snapshot_id": f"local-{content_id[:16]}", "sources": entries}
    path = root / "manifests" / f"{content_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_manifest(root, path)
    return path
