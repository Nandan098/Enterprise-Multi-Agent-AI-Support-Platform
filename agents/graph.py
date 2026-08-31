from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from llm_client import generate_text
from rag.retriever import retrieve_with_scores
from tools.actions import create_support_ticket


class AgentState(TypedDict, total=False):
    question: str
    route: str
    context: str
    sources: list[dict]
    retrieval: list[dict]
    draft: str
    validation: str
    validation_reason: str
    action: dict
    answer: str


def _clean_label(text: str) -> str:
    label = text.strip().upper().splitlines()[0] if text.strip() else "KNOWLEDGE"
    for candidate in ("KNOWLEDGE", "SUPPORT", "ACTION"):
        if candidate in label:
            return candidate
    return "KNOWLEDGE"


def router(state: AgentState) -> AgentState:
    prompt = """Classify the user request as exactly one of: KNOWLEDGE, SUPPORT, ACTION.
KNOWLEDGE = information or policy questions.
SUPPORT = troubleshooting or guidance.
ACTION = explicitly asks to create/update a ticket or take an external action.
Return only the label."""
    route = _clean_label(generate_text(prompt + "\nUser: " + state["question"]))
    return {**state, "route": route}


def _build_context(question: str):
    hits = retrieve_with_scores(question)
    context_parts: list[str] = []
    sources: list[dict] = []
    for hit in hits:
        context_parts.append(
            f"[SOURCE: {hit['source']} | PAGE: {hit['page']} | CHUNK: {hit['chunk_id']}]\n{hit['content']}"
        )
        sources.append({k: hit[k] for k in ("source", "page", "chunk_id", "distance")})
    return "\n\n---\n\n".join(context_parts), sources, hits


def knowledge_agent(state: AgentState) -> AgentState:
    context, sources, retrieval = _build_context(state["question"])
    if not context.strip():
        return {
            **state,
            "context": "",
            "sources": [],
            "retrieval": [],
            "draft": "I could not find indexed evidence for that request.",
        }
    prompt = f"""You are an enterprise knowledge agent. Answer ONLY from the supplied context.
Never invent policy, names, dates, numbers, or procedures. Cite used facts inline as [source, page].
If the context is insufficient, say so.

CONTEXT:\n{context}\n\nUSER:\n{state['question']}"""
    response = generate_text(prompt)
    return {
        **state,
        "context": context,
        "sources": sources,
        "retrieval": retrieval,
        "draft": response,
    }


def support_agent(state: AgentState) -> AgentState:
    result = knowledge_agent(state)
    result["draft"] = "Troubleshooting guidance based on the indexed enterprise knowledge base:\n\n" + result.get("draft", "")
    return result


def action_agent(state: AgentState) -> AgentState:
    ticket = create_support_ticket(state["question"])
    return {
        **state,
        "action": ticket,
        "draft": f"Created support ticket {ticket['ticket_id']} with status {ticket['status']}.",
        "sources": [],
        "retrieval": [],
    }


def validator(state: AgentState) -> AgentState:
    if state.get("route") == "ACTION":
        return {
            **state,
            "validation": "PASS",
            "validation_reason": "Response is based on a controlled tool result.",
            "answer": state.get("draft", ""),
        }
    context = state.get("context", "")
    draft = state.get("draft", "")
    if not context.strip():
        return {
            **state,
            "validation": "REVISE",
            "validation_reason": "No indexed evidence was available.",
            "answer": "I could not confidently answer because the knowledge base has no indexed evidence for this request.",
        }
    prompt = f"""You are a strict grounding validator. Decide whether the ANSWER is supported by the CONTEXT.
Return exactly two lines:
PASS
Reason: <one sentence>
OR
REVISE
Reason: <one sentence>

CONTEXT:\n{context}\n\nANSWER:\n{draft}"""
    verdict = generate_text(prompt)
    lines = verdict.splitlines()
    status = lines[0].strip().upper() if lines else "REVISE"
    status = status if status in {"PASS", "REVISE"} else "REVISE"
    reason = lines[1].replace("Reason:", "").strip() if len(lines) > 1 else "Validator did not provide a reason."
    answer = draft if status == "PASS" else "I could not confidently ground this answer in the indexed documents. Please upload the relevant policy/SOP and try again."
    return {**state, "validation": status, "validation_reason": reason, "answer": answer}


def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in [
        ("router", router),
        ("knowledge", knowledge_agent),
        ("support", support_agent),
        ("action", action_agent),
        ("validator", validator),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"].lower(),
        {"knowledge": "knowledge", "support": "support", "action": "action"},
    )
    for node in ("knowledge", "support", "action"):
        graph.add_edge(node, "validator")
    graph.add_edge("validator", END)
    return graph.compile()


APP = build_graph()


def run_agent(question: str) -> dict:
    result = APP.invoke({"question": question})
    return {
        "answer": result.get("answer", result.get("draft", "")),
        "route": result.get("route", ""),
        "validation": result.get("validation", ""),
        "validation_reason": result.get("validation_reason", ""),
        "sources": result.get("sources", []),
        "retrieval": result.get("retrieval", []),
        "action": result.get("action"),
    }
