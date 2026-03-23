"""Unit tests for BrowserManager — filesystem logic, no real browser."""

from __future__ import annotations

import os
import time

from rainier.scrapers.browser import BrowserManager


class TestIsSessionValid:
    def test_no_file(self, tmp_path):
        assert BrowserManager.is_session_valid(tmp_path / "nope.json", ttl_hours=12) is False

    def test_fresh_file(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        assert BrowserManager.is_session_valid(p, ttl_hours=12) is True

    def test_expired_file(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        # Backdate file modification time by 13 hours
        old_time = time.time() - (13 * 3600)
        os.utime(p, (old_time, old_time))
        assert BrowserManager.is_session_valid(p, ttl_hours=12) is False

    def test_exactly_at_boundary(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        # Set to exactly 12 hours ago — should be invalid (not strictly less than)
        boundary_time = time.time() - (12 * 3600)
        os.utime(p, (boundary_time, boundary_time))
        assert BrowserManager.is_session_valid(p, ttl_hours=12) is False

    def test_just_under_boundary(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        # Set to 11 hours 59 minutes ago — should be valid
        under_time = time.time() - (11 * 3600 + 59 * 60)
        os.utime(p, (under_time, under_time))
        assert BrowserManager.is_session_valid(p, ttl_hours=12) is True

    def test_zero_ttl(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{}")
        assert BrowserManager.is_session_valid(p, ttl_hours=0) is False
