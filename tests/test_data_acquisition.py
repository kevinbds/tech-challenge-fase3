import json

import pytest

from alfabetizacao_ml.data import acquisition


def _fixture_source(monkeypatch, payload=b"synthetic-public-test-source"):
    monkeypatch.setattr(acquisition, "SOURCES", [{
        "id": "synthetic_source", "url": "https://example.invalid/fixture.csv",
        "filename": "fixture.csv", "observation_year": 2024, "kind": "synthetic",
    }])

    def download(url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return {"download_url": url, "transport": "https"}

    monkeypatch.setattr(acquisition, "_download", download)


def test_frozen_replay_uses_hashes_not_clock_or_network(tmp_path, monkeypatch):
    _fixture_source(monkeypatch)
    manifest = acquisition.acquire_sources(tmp_path)
    record = json.loads(manifest.read_text())
    record["sources"][0]["acquired_at"] = "2000-01-01T00:00:00+00:00"
    manifest.write_text(json.dumps(record))
    monkeypatch.setattr(acquisition, "_download", lambda *_: pytest.fail("Intact replay contacted network"))
    assert acquisition.acquire_sources(tmp_path, "reproduce", manifest) == manifest


def test_changed_cached_byte_blocks_replay(tmp_path, monkeypatch):
    _fixture_source(monkeypatch)
    manifest = acquisition.acquire_sources(tmp_path)
    entry = json.loads(manifest.read_text())["sources"][0]
    (tmp_path / entry["relative_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity mismatch"):
        acquisition.acquire_sources(tmp_path, "reproduce", manifest)


def test_changed_remote_source_does_not_replace_frozen_input(tmp_path, monkeypatch):
    _fixture_source(monkeypatch)
    manifest = acquisition.acquire_sources(tmp_path)
    entry = json.loads(manifest.read_text())["sources"][0]
    (tmp_path / entry["relative_path"]).unlink()
    _fixture_source(monkeypatch, b"revised-official-fixture")
    with pytest.raises(ValueError, match="Remote source changed"):
        acquisition.acquire_sources(tmp_path, "reproduce", manifest)


def test_update_creates_new_content_address_without_overwriting_old(tmp_path, monkeypatch):
    _fixture_source(monkeypatch)
    old = acquisition.acquire_sources(tmp_path)
    _fixture_source(monkeypatch, b"new-version")
    new = acquisition.acquire_sources(tmp_path)
    assert old != new
    acquisition.verify_manifest(tmp_path, old)
    acquisition.verify_manifest(tmp_path, new)


def test_manifest_path_cannot_escape_cache(tmp_path, monkeypatch):
    _fixture_source(monkeypatch)
    manifest = acquisition.acquire_sources(tmp_path)
    record = json.loads(manifest.read_text())
    record["sources"][0]["relative_path"] = "../outside"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="escapes"):
        acquisition.acquire_sources(tmp_path, "reproduce", manifest)


def test_reproduction_requires_manifest(tmp_path):
    with pytest.raises(ValueError, match="expected manifest"):
        acquisition.acquire_sources(tmp_path, "reproduce")


@pytest.mark.parametrize("mutation", ["url", "id", "duplicate", "snapshot", "kind"])
def test_tampered_metadata_rejected_before_network(tmp_path, monkeypatch, mutation):
    _fixture_source(monkeypatch)
    manifest = acquisition.acquire_sources(tmp_path)
    record = json.loads(manifest.read_text())
    entry = record["sources"][0]
    (tmp_path / entry["relative_path"]).unlink()
    if mutation == "duplicate":
        record["sources"].append(entry.copy())
    elif mutation == "snapshot":
        record["snapshot_id"] = "../../outside"
    else:
        entry[mutation] = "untrusted"
    manifest.write_text(json.dumps(record))
    monkeypatch.setattr(acquisition, "_download", lambda *_: pytest.fail("Unvalidated manifest contacted network"))
    with pytest.raises(ValueError):
        acquisition.acquire_sources(tmp_path, "reproduce", manifest)
