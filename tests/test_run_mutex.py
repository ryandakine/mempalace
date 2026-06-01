"""Plan §6: whole-run flock — 2nd concurrent run exits immediately."""

import pytest

from mempalace.memory_miner.watermark import AlreadyRunning, run_lock


def test_second_concurrent_lock_raises(tmp_path):
    lock = tmp_path / "miner.lock"
    with run_lock(lock):
        # while the first lock is held, a second acquisition must fail fast.
        with pytest.raises(AlreadyRunning):
            with run_lock(lock):
                pass


def test_lock_released_after_context(tmp_path):
    lock = tmp_path / "miner.lock"
    with run_lock(lock):
        pass
    # released — can re-acquire.
    with run_lock(lock):
        pass


def test_lock_writes_pid(tmp_path):
    lock = tmp_path / "miner.lock"
    with run_lock(lock):
        content = lock.read_text().strip()
        assert content.isdigit()
