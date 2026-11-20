from fastapi.testclient import TestClient

from src.api import MAX_BATCH, app

client = TestClient(app)

VISIT = {
    "session_id": "new-1", "client_id": "visitor-42", "visit_dt": "2021-11-09T10:15:00", "visit_number": 1,
    "utm_source": "src_a", "utm_medium": "cpc", "utm_campaign": "camp_1", "device_category": "mobile",
    "device_os": "Android", "device_screen_resolution": "360x720", "device_browser": "Chrome",
    "geo_country": "Russia", "geo_city": "Moscow", "landing_path": "sberauto.com/cars/bmw",
    "first_events": ["view_card", "photos"],
}


def test_health(store):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"]


def test_scores_a_raw_visit(store):
    r = client.post("/score/early-session", json=[VISIT])
    assert r.status_code == 200
    out = r.json()[0]
    assert 0 <= out["probability"] <= 1 and isinstance(out["priority"], bool)


def test_finished_visit_becomes_history(store):
    first = client.post("/score/early-session", json=[VISIT]).json()[0]["probability"]
    r = client.post("/visits/finished", json={"client_id": "visitor-42", "visit_dt": "2021-11-09T10:15:00",
                                              "utm_medium": "cpc", "n_hits": 25,
                                              "events": ["view_card"] * 5 + ["sub_car_claim_click"]})
    assert r.status_code == 204
    again = {**VISIT, "session_id": "new-2", "visit_dt": "2021-11-09T18:00:00", "visit_number": 2}
    second = client.post("/score/early-session", json=[again]).json()[0]["probability"]
    assert second != first      # the earlier visit, with a conversion, is now part of the history


def test_conversion_in_first_events_is_not_scored(store):
    done = {**VISIT, "first_events": ["view_card", "sub_car_claim_click"]}
    out = client.post("/score/early-session", json=[done]).json()[0]
    assert out["probability"] is None and out["priority"] is False


def test_rejects_bad_input(store):
    no_events = {k: v for k, v in VISIT.items() if k != "first_events"}
    assert client.post("/score/early-session", json=[no_events]).status_code == 422
    too_many = {**VISIT, "first_events": ["a", "b", "c", "d"]}
    assert client.post("/score/early-session", json=[too_many]).status_code == 422


def test_rejects_oversized_batch(store):
    assert client.post("/score/early-session", json=[VISIT] * (MAX_BATCH + 1)).status_code == 413


def test_missing_store_returns_503(tmp_path, monkeypatch):
    from src import api
    monkeypatch.setattr(api, "STORE", tmp_path)
    assert client.post("/score/early-session", json=[VISIT]).status_code == 503
