from pydantic import BaseModel

class TicketIn(BaseModel):
    text: str
    product: str | None = None
    channel: str | None = None
    fcr: int | None = None
    kb_used: str | None = None

class PredictionOut(BaseModel):
    labels: dict
    feedback_bullets: list[str]