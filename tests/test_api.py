import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_create_student():
    resp = client.post("/students", json={"name": "Alice"})
    assert resp.status_code in (200, 201)
    data = resp.json()
    assert "id" in data or "student_id" in data


def test_list_students():
    resp = client.get("/students")
    assert resp.status_code == 200
    data = resp.json()
    if isinstance(data, list):
        assert len(data) >= 0
    else:
        assert "items" in data or "students" in data or "data" in data


def test_delete_student():
    create = client.post("/students", json={"name": "Bob"})
    assert create.status_code in (200, 201)
    created = create.json()
    sid = created.get("id") or created.get("student_id")
    assert sid is not None

    resp = client.delete(f"/students/{sid}")
    assert resp.status_code in (200, 204)
