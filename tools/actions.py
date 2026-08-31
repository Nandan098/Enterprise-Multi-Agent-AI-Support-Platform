from __future__ import annotations

import uuid


def create_support_ticket(issue: str) -> dict[str, str]:
    """Controlled demo tool. No external side effect occurs in this portfolio build."""
    return {
        "ticket_id": f"DEMO-{uuid.uuid4().hex[:8].upper()}",
        "status": "CREATED",
        "priority": "NORMAL",
        "issue": issue,
    }
