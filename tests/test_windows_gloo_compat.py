from __future__ import annotations

from raylight.distributed_worker.windows_gloo import (
    BACKEND_NAME,
    install_windows_gloo_new_group_router,
)


class FakeDistributed:
    def __init__(self):
        self.calls = []

    def new_group(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "group"


def test_new_group_router_replaces_hardcoded_gloo_keyword():
    fake = FakeDistributed()

    original = install_windows_gloo_new_group_router(fake)

    assert original is not fake.new_group
    assert fake.new_group([0, 1], backend="gloo") == "group"
    assert fake.calls == [(([0, 1],), {"backend": BACKEND_NAME})]


def test_new_group_router_replaces_hardcoded_gloo_positional_backend():
    fake = FakeDistributed()
    install_windows_gloo_new_group_router(fake)

    fake.new_group([0, 1], None, "gloo")

    assert fake.calls == [(([0, 1], None, BACKEND_NAME), {})]


def test_new_group_router_is_idempotent_and_preserves_other_backends():
    fake = FakeDistributed()

    first_original = install_windows_gloo_new_group_router(fake)
    second_original = install_windows_gloo_new_group_router(fake)
    fake.new_group([0, 1], backend=BACKEND_NAME)

    assert second_original is first_original
    assert fake.calls == [(([0, 1],), {"backend": BACKEND_NAME})]
