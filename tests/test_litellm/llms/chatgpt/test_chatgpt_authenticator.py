import base64
import json
import os
import stat
import threading
import time
from unittest.mock import patch

import httpx
import pytest

from litellm.llms.chatgpt.authenticator import (
    Authenticator,
    get_cached_authenticator,
    get_chatgpt_auth_file,
)
from litellm.types.router import GenericLiteLLMParams


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


class TestChatGPTAuthenticator:
    @pytest.fixture
    def authenticator(self, tmp_path):
        return Authenticator(auth_file=str(tmp_path / "auth.json"))

    def test_get_access_token_from_file(self, authenticator):
        future_time = time.time() + 3600
        with open(authenticator.auth_file, "w") as auth_file:
            json.dump({"access_token": "token-123", "expires_at": future_time}, auth_file)

        token = authenticator.get_access_token()

        assert token == "token-123"

    def test_get_access_token_refresh(self, authenticator):
        past_time = time.time() - 10
        with open(authenticator.auth_file, "w") as auth_file:
            json.dump(
                {
                    "access_token": "token-old",
                    "refresh_token": "refresh-123",
                    "expires_at": past_time,
                },
                auth_file,
            )
        refreshed = {
            "access_token": "token-new",
            "refresh_token": "refresh-123",
            "id_token": "id-123",
        }

        with patch.object(authenticator, "_refresh_tokens", return_value=refreshed):
            token = authenticator.get_access_token()

        assert token == "token-new"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}})
        with open(authenticator.auth_file, "w") as auth_file:
            json.dump({"id_token": id_token}, auth_file)

        with patch.object(authenticator, "_write_auth_file") as mock_write:
            account_id = authenticator.get_account_id()

        assert account_id == "acct-123"
        mock_write.assert_called_once()
        assert mock_write.call_args[0][0]["account_id"] == "acct-123"


def _write_auth_record(path, token: str, account_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access_token": token,
                "account_id": account_id,
                "expires_at": time.time() + 3600,
            }
        )
    )


class TestChatGPTMultiAccountAuthenticator:
    def test_explicit_auth_file_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "env-dir"))
        monkeypatch.setenv("CHATGPT_AUTH_FILE", "env.json")
        custom_file = tmp_path / "account-a" / "auth.json"

        authenticator = Authenticator(auth_file=str(custom_file))

        assert authenticator.auth_file == str(custom_file)
        assert authenticator.token_dir == str(custom_file.parent)
        assert custom_file.parent.exists()

    def test_default_auth_file_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "env-dir"))
        monkeypatch.setenv("CHATGPT_AUTH_FILE", "env.json")

        authenticator = Authenticator()

        assert authenticator.auth_file == str(tmp_path / "env-dir" / "env.json")

    def test_relative_auth_file_uses_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        authenticator = Authenticator(auth_file="account-a/auth.json")

        assert authenticator.auth_file == str(tmp_path / "account-a" / "auth.json")
        assert authenticator.token_dir == str(tmp_path / "account-a")
        assert (tmp_path / "account-a").is_dir()

    def test_auth_file_expands_home_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        authenticator = Authenticator(auth_file="~/.config/litellm/account-a.json")

        assert authenticator.auth_file == str(tmp_path / ".config" / "litellm" / "account-a.json")

    def test_two_auth_files_are_isolated(self, tmp_path):
        file_a = tmp_path / "account-a.json"
        file_b = tmp_path / "account-b.json"
        _write_auth_record(file_a, "token-a", "acct-a")
        _write_auth_record(file_b, "token-b", "acct-b")

        authenticator_a = Authenticator(auth_file=str(file_a))
        authenticator_b = Authenticator(auth_file=str(file_b))

        assert authenticator_a.get_access_token() == "token-a"
        assert authenticator_a.get_account_id() == "acct-a"
        assert authenticator_b.get_access_token() == "token-b"
        assert authenticator_b.get_account_id() == "acct-b"

    def test_get_cached_authenticator_reuses_instance_per_path(self, tmp_path):
        file_a = tmp_path / "account-a.json"
        file_b = tmp_path / "account-b.json"

        authenticator_a = get_cached_authenticator(str(file_a))

        assert get_cached_authenticator(str(file_a)) is authenticator_a
        assert get_cached_authenticator(str(file_b)) is not authenticator_a

    def test_get_cached_authenticator_reuses_instance_for_equivalent_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        absolute_path = tmp_path / "account-a" / "auth.json"

        authenticator = get_cached_authenticator("account-a/auth.json")

        assert get_cached_authenticator(str(absolute_path)) is authenticator

    def test_get_chatgpt_auth_file(self):
        assert get_chatgpt_auth_file(None) is None
        assert get_chatgpt_auth_file({}) is None
        assert get_chatgpt_auth_file({"chatgpt_auth_file": ""}) is None
        assert get_chatgpt_auth_file({"chatgpt_auth_file": "/a/auth.json"}) == "/a/auth.json"
        assert get_chatgpt_auth_file(GenericLiteLLMParams()) is None
        assert get_chatgpt_auth_file(GenericLiteLLMParams(chatgpt_auth_file="/b/auth.json")) == "/b/auth.json"

    def test_atomic_write_uses_private_mode_and_complete_json(self, tmp_path):
        auth_file = tmp_path / "account-a" / "auth.json"
        authenticator = Authenticator(auth_file=str(auth_file))
        auth_data = {
            "access_token": "new-token",
            "account_id": "acct-a",
            "expires_at": time.time() + 3600,
        }

        authenticator._write_auth_file(auth_data)

        assert json.loads(auth_file.read_text()) == auth_data
        assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600
        assert not list(auth_file.parent.glob(".auth.json.*"))

    def test_failed_atomic_write_preserves_existing_file(self, tmp_path):
        auth_file = tmp_path / "account-a" / "auth.json"
        authenticator = Authenticator(auth_file=str(auth_file))
        old_data = {"access_token": "old-token", "expires_at": time.time() + 3600}
        authenticator._write_auth_file(old_data)

        with patch("os.replace", side_effect=OSError("synthetic failure")):
            write_result = authenticator._write_auth_file({"access_token": "new-token"})

        assert write_result is None
        assert json.loads(auth_file.read_text()) == old_data
        assert not list(auth_file.parent.glob(".auth.json.*"))

    def test_cached_authenticator_observes_atomic_replacement(self, tmp_path):
        auth_file = tmp_path / "account-a" / "auth.json"
        _write_auth_record(auth_file, "token-old", "acct-a")
        authenticator = get_cached_authenticator(str(auth_file))
        assert authenticator.get_access_token() == "token-old"

        replacement = auth_file.parent / "auth.json.next"
        _write_auth_record(replacement, "token-new", "acct-a")
        os.replace(replacement, auth_file)

        assert authenticator.get_access_token() == "token-new"

    def test_refresh_does_not_overwrite_imported_replacement(self, tmp_path):
        auth_file = tmp_path / "account-a" / "auth.json"
        auth_file.parent.mkdir(parents=True)
        auth_file.write_text(
            json.dumps(
                {
                    "access_token": "token-old",
                    "refresh_token": "refresh-old",
                    "account_id": "acct-a",
                    "expires_at": time.time() - 10,
                }
            )
        )
        authenticator = Authenticator(auth_file=str(auth_file))

        def refresh_and_replace(url, **kwargs):
            replacement = auth_file.parent / "auth.json.next"
            _write_auth_record(replacement, "token-imported", "acct-a")
            os.replace(replacement, auth_file)
            return httpx.Response(
                200,
                json={
                    "access_token": "token-refreshed-from-old-session",
                    "refresh_token": "refresh-old",
                    "id_token": "id-old",
                },
                request=httpx.Request("POST", url),
            )

        with patch("litellm.llms.chatgpt.authenticator._get_httpx_client") as get_client:
            get_client.return_value.post.side_effect = refresh_and_replace
            access_token = authenticator.get_access_token()

        assert access_token == "token-imported"
        assert json.loads(auth_file.read_text())["access_token"] == "token-imported"
        assert not list(auth_file.parent.glob(".auth.json.*"))

    @pytest.mark.skipif(os.name != "posix", reason="requires POSIX flock")
    def test_read_waits_for_exclusive_import_lock(self, tmp_path):
        import fcntl

        auth_file = tmp_path / "account-a" / "auth.json"
        _write_auth_record(auth_file, "token-a", "acct-a")
        authenticator = Authenticator(auth_file=str(auth_file))
        lock_file = auth_file.parent / ".import.lock"
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
        started = threading.Event()
        finished = threading.Event()
        result = []

        def read_auth_file():
            started.set()
            result.append(authenticator._read_auth_file())
            finished.set()

        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            thread = threading.Thread(target=read_auth_file)
            thread.start()
            assert started.wait(timeout=1)
            assert not finished.wait(timeout=0.1)
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            thread.join(timeout=1)
        finally:
            os.close(lock_fd)

        assert finished.is_set()
        assert result[0]["access_token"] == "token-a"
