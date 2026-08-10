"""Structured, replaceable natural-language capability selector."""

from __future__ import annotations

from dataclasses import dataclass, field


AVAILABLE_CAPABILITIES = (
    "place_bottle_on_shelf",
    "pull_tissue_from_box",
    "draw_horizontal_line",
)


@dataclass(frozen=True)
class SkillRequest:
    skill: str
    args: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"skill": self.skill, "args": self.args}


@dataclass(frozen=True)
class AgentResponse:
    status: str
    request: SkillRequest | None
    available_capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "request": self.request.to_dict() if self.request else None,
            "available_capabilities": list(self.available_capabilities),
        }


class MockAgent:
    """Keyword router implementing the same boundary expected of an LLMAgent."""

    def select(self, command: str) -> AgentResponse:
        normalized = command.casefold().strip()
        bottle_terms = ("bottle", "瓶", "水瓶")
        shelf_terms = ("shelf", "rack", "架", "架子")
        if any(term in normalized for term in bottle_terms) and any(
            term in normalized for term in shelf_terms
        ):
            return AgentResponse(
                status="OK",
                request=SkillRequest(skill="place_bottle_on_shelf", args={}),
                available_capabilities=AVAILABLE_CAPABILITIES,
            )
        tissue_terms = ("tissue", "纸巾", "抽纸", "纸")
        pull_terms = ("pull", "extract", "抽", "拉出", "取出")
        if any(term in normalized for term in tissue_terms) and any(
            term in normalized for term in pull_terms
        ):
            return AgentResponse(
                status="OK",
                request=SkillRequest(skill="pull_tissue_from_box", args={}),
                available_capabilities=AVAILABLE_CAPABILITIES,
            )
        pen_terms = ("pen", "笔", "marker")
        draw_terms = ("draw", "画", "横线", "一横", "line")
        fixed_line_terms = ("横线", "一横", "horizontal line")
        if (
            any(term in normalized for term in pen_terms)
            and any(term in normalized for term in draw_terms)
        ) or any(term in normalized for term in fixed_line_terms):
            return AgentResponse(
                status="OK",
                request=SkillRequest(skill="draw_horizontal_line", args={}),
                available_capabilities=AVAILABLE_CAPABILITIES,
            )
        return AgentResponse(
            status="CANNOT_EXECUTE",
            request=None,
            available_capabilities=AVAILABLE_CAPABILITIES,
        )
