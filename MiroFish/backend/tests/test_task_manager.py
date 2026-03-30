"""Tests for TaskManager and Task model."""

import pytest
import threading
from app.models.task import TaskManager, TaskStatus


@pytest.fixture(autouse=True)
def reset_task_manager():
    """Reset singleton state between tests."""
    mgr = TaskManager()
    mgr._tasks.clear()
    yield mgr


class TestTaskManager:
    def test_create_task(self, reset_task_manager):
        mgr = reset_task_manager
        task_id = mgr.create_task("graph_build", {"project_id": "p1"})
        assert task_id is not None

        task = mgr.get_task(task_id)
        assert task is not None
        assert task.task_type == "graph_build"
        assert task.status == TaskStatus.PENDING
        assert task.metadata == {"project_id": "p1"}

    def test_update_task(self, reset_task_manager):
        mgr = reset_task_manager
        task_id = mgr.create_task("simulation")
        mgr.update_task(task_id, status=TaskStatus.PROCESSING, progress=50, message="halfway")

        task = mgr.get_task(task_id)
        assert task.status == TaskStatus.PROCESSING
        assert task.progress == 50
        assert task.message == "halfway"

    def test_complete_task(self, reset_task_manager):
        mgr = reset_task_manager
        task_id = mgr.create_task("report")
        mgr.complete_task(task_id, result={"report_id": "r1"})

        task = mgr.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.progress == 100
        assert task.result == {"report_id": "r1"}

    def test_fail_task(self, reset_task_manager):
        mgr = reset_task_manager
        task_id = mgr.create_task("simulation")
        mgr.fail_task(task_id, error="LLM timeout")

        task = mgr.get_task(task_id)
        assert task.status == TaskStatus.FAILED
        assert task.error == "LLM timeout"

    def test_get_nonexistent_task(self, reset_task_manager):
        mgr = reset_task_manager
        assert mgr.get_task("nonexistent-id") is None

    def test_list_tasks(self, reset_task_manager):
        mgr = reset_task_manager
        mgr.create_task("graph_build")
        mgr.create_task("simulation")
        mgr.create_task("graph_build")

        all_tasks = mgr.list_tasks()
        assert len(all_tasks) == 3

        graph_tasks = mgr.list_tasks(task_type="graph_build")
        assert len(graph_tasks) == 2

    def test_task_to_dict(self, reset_task_manager):
        mgr = reset_task_manager
        task_id = mgr.create_task("report")
        task = mgr.get_task(task_id)
        d = task.to_dict()

        assert d["task_id"] == task_id
        assert d["task_type"] == "report"
        assert d["status"] == "pending"
        assert "created_at" in d
        assert "updated_at" in d

    def test_cleanup_old_tasks(self, reset_task_manager):
        mgr = reset_task_manager
        tid1 = mgr.create_task("old_task")
        tid2 = mgr.create_task("new_task")

        mgr.complete_task(tid1, result={})
        # Force old timestamp
        from datetime import datetime, timedelta
        mgr._tasks[tid1].created_at = datetime.now() - timedelta(hours=48)

        mgr.cleanup_old_tasks(max_age_hours=24)
        assert mgr.get_task(tid1) is None
        assert mgr.get_task(tid2) is not None

    def test_thread_safety(self, reset_task_manager):
        mgr = reset_task_manager
        ids = []

        def create_tasks():
            for _ in range(50):
                tid = mgr.create_task("concurrent")
                ids.append(tid)

        threads = [threading.Thread(target=create_tasks) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(mgr.list_tasks()) == 200
