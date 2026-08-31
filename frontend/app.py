from __future__ import annotations

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    BACKEND_URL = st.secrets.get("BACKEND_URL", os.getenv("BACKEND_URL", "http://localhost:8000"))
except Exception:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise Multi-Agent AI Support Platform")
st.caption("RAG • LangGraph agents • grounding validation • controlled actions")

# ---- Backend status / demo setup ----
with st.sidebar:
    st.header("🚀 Live Demo")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if health.ok:
            st.success("Knowledge base ready")
        else:
            st.warning("Backend responded with an error")
    except requests.RequestException:
        st.error("Backend unavailable")

    st.caption("Demo knowledge base is preloaded. Just ask a question.")
    with st.expander("Add your own PDF (optional)"):
        uploaded = st.file_uploader("Upload enterprise PDF", type=["pdf"])
        if uploaded and st.button("Index document"):
            with st.spinner("Indexing..."):
                try:
                    r = requests.post(
                        f"{BACKEND_URL}/ingest",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=180,
                    )
                    if r.ok:
                        st.success(r.json())
                    else:
                        st.error(r.text)
                except requests.RequestException as exc:
                    st.error(f"Backend connection failed: {exc}")

    st.divider()
    st.header("Agent Flow")
    st.code("Router → Knowledge/Support/Action → Validator", language="text")
    st.caption("Designed for an enterprise AI support workflow.")

question = st.text_area(
    "Ask an enterprise question",
    height=110,
    placeholder="What is the password reset policy?",
)

if st.button("Ask AI", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Agents are working..."):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": question},
                    timeout=180,
                )
            except requests.RequestException as exc:
                r = None
                st.error(f"Backend connection failed: {exc}")

        if r is not None:
            if r.ok:
                d = r.json()
                a, b, c, e = st.columns(4)
                a.metric("Route", d.get("route", ""))
                b.metric("Validation", d.get("validation", ""))
                c.metric("Retrieved", d.get("retrieval_count", 0))
                e.metric("Latency", f"{d.get('latency_ms', 0)} ms")

                st.subheader("Answer")
                st.write(d.get("answer", ""))

                if d.get("validation_reason"):
                    st.info("Validator: " + d["validation_reason"])

                if d.get("sources"):
                    st.subheader("Grounding sources")
                    for s in d["sources"]:
                        st.write(
                            f"📄 **{s['source']}** — page {s['page']} • "
                            f"chunk `{s['chunk_id']}` • distance `{s['distance']}`"
                        )
                    with st.expander("Show retrieved excerpts"):
                        for h in d.get("retrieval", []):
                            st.markdown(
                                f"**{h['source']} — page {h['page']}**\n\n{h['content']}"
                            )

                if d.get("action"):
                    st.subheader("Controlled action result")
                    st.json(d["action"])
            else:
                st.error(r.text)

st.divider()
st.subheader("🧪 Quick Evaluation")
st.caption("Measure retrieval/grounding behavior for a question.")
q = st.text_input("Evaluation question", placeholder="What is the password reset policy?")
keys = st.text_input("Expected keywords", placeholder="password, reset, 90")
if st.button("Run evaluation"):
    if not q.strip():
        st.warning("Enter an evaluation question.")
    else:
        keywords = [x.strip() for x in keys.split(",") if x.strip()]
        try:
            r = requests.post(
                f"{BACKEND_URL}/evaluate",
                json={"question": q, "expected_keywords": keywords},
                timeout=180,
            )
            if r.ok:
                d = r.json()
                x, y, z = st.columns(3)
                x.metric(
                    "Keyword score",
                    "N/A" if d.get("keyword_score") is None else d["keyword_score"],
                )
                y.metric("Validation", d.get("validation", ""))
                z.metric("Latency", f"{d.get('latency_ms', 0)} ms")
                st.write("Matched keywords:", d.get("matched_keywords", []))
            else:
                st.error(r.text)
        except requests.RequestException as exc:
            st.error(f"Backend connection failed: {exc}")
