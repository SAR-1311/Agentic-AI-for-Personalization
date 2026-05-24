"""Streamlit chat UI with three tabs:
  1. Chat            — converse with the agent, see retrieved memories.
  2. Persona Inspector — live view of the structured persona profile.
  3. Memory Manager  — table of LTM with revoke buttons + decay sweep trigger.

Run:
    streamlit run frontend/app.py
"""
import os
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic Memory AI", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------- helpers

def post(path, **kw):
    return requests.post(f"{BACKEND}{path}", **kw, timeout=120)

def get(path, **kw):
    return requests.get(f"{BACKEND}{path}", **kw, timeout=30)

def delete(path, **kw):
    return requests.delete(f"{BACKEND}{path}", **kw, timeout=30)


# ---------------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🧠 Agentic Memory AI")
    st.markdown("*Long-term personalised assistant.*")
    user_id = st.text_input("User ID", value="demo_user")
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if st.button("🆕 New session"):
        st.session_state.messages = []
        st.session_state.session_id = None
        if user_id:
            try:
                delete(f"/session/{user_id}")
            except Exception:
                pass
        st.rerun()

    st.divider()
    st.subheader("Forgetting")
    if st.button("🌀 Run decay sweep"):
        try:
            r = post(f"/forgetting/run", params={"user_id": user_id})
            st.success(r.json())
        except Exception as e:
            st.error(str(e))


# ---------------------------------------------------------------------- tabs

tab_chat, tab_persona, tab_memory = st.tabs(["💬 Chat", "🪪 Persona Inspector", "📚 Memory Manager"])


# ============================================================ CHAT
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "diagnostics" not in st.session_state:
        st.session_state.diagnostics = []

    col_chat, col_diag = st.columns([2, 1])

    with col_chat:
        st.subheader("Conversation")
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["text"])

        if prompt := st.chat_input("Say something to the agent…"):
            st.session_state.messages.append({"role": "user", "text": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                r = post("/chat", json={
                    "user_id": user_id, "message": prompt,
                    "session_id": st.session_state.session_id,
                })
                r.raise_for_status()
                data = r.json()
                st.session_state.session_id = data["session_id"]
                reply = data["reply"]
                st.session_state.messages.append({"role": "assistant", "text": reply})
                st.session_state.diagnostics.append(data["diagnostics"])
                with st.chat_message("assistant"):
                    st.markdown(reply)
            except Exception as e:
                st.error(f"Backend error: {e}")

    with col_diag:
        st.subheader("Last-turn diagnostics")
        if st.session_state.diagnostics:
            d = st.session_state.diagnostics[-1]
            with st.expander("🚪 Gatekeeper", expanded=True):
                gk = d.get("gatekeeper", {})
                w = gk.get("weights", {})
                st.caption(f"α={w.get('alpha')} β={w.get('beta')} γ={w.get('gamma')} τ={w.get('threshold')}")
                if gk.get("passed"):
                    st.markdown("**Passed (high signal):**")
                    st.dataframe(gk["passed"], use_container_width=True, hide_index=True)
                if gk.get("rejected"):
                    st.markdown("**Rejected (noise):**")
                    st.dataframe(gk["rejected"], use_container_width=True, hide_index=True)

            with st.expander("🧬 Synthesised traits"):
                st.write(d.get("synthesized") or "—")

            with st.expander("📚 Retrieved memories"):
                st.dataframe(d.get("retrieved") or [], use_container_width=True, hide_index=True)

            st.metric("Persona size", d.get("persona_size", 0))


# ============================================================ PERSONA
with tab_persona:
    st.subheader(f"Living Persona — {user_id}")
    show_history = st.toggle("Show full history (incl. superseded)", value=False)
    try:
        r = get(f"/persona/{user_id}", params={"history": show_history})
        data = r.json()
        st.code(data.get("summary", ""), language="markdown")
        traits = data.get("traits", [])
        if traits:
            st.dataframe(traits, use_container_width=True, hide_index=True)
        else:
            st.info("No persona traits yet — chat a bit first.")
    except Exception as e:
        st.error(f"Backend error: {e}")


# ============================================================ MEMORY MANAGER
with tab_memory:
    st.subheader(f"Long-Term Memory — {user_id}")
    include_pruned = st.toggle("Include pruned/superseded", value=False)
    try:
        r = get(f"/memory/{user_id}", params={"include_pruned": include_pruned})
        data = r.json()
        memories = data.get("memories", [])
        if not memories:
            st.info("No memories stored yet.")
        else:
            st.write(f"**{len(memories)} memories**")
            for m in sorted(memories, key=lambda x: x.get("decayed_importance", 0), reverse=True):
                with st.container(border=True):
                    cols = st.columns([4, 1, 1])
                    cols[0].markdown(f"**[{m.get('trait_type', '?')}]** {m.get('text', '')}")
                    cols[0].caption(
                        f"I₀={m.get('importance', 0):.2f} · "
                        f"I_t={m.get('decayed_importance', 0):.2f} · "
                        f"f={m.get('frequency', 0):.2f} · "
                        f"c={m.get('confidence', 0):.2f} · "
                        f"e={m.get('emotion', 0):.2f}"
                    )
                    if cols[1].button("Revoke", key=f"rev_{m['id']}"):
                        delete(f"/memory/{m['id']}", params={"hard": True})
                        st.rerun()

        st.divider()
        st.subheader("Revoke a whole cluster")
        trait_type = st.selectbox("Trait type", [
            "preference", "dietary", "occupation", "health",
            "relationship", "goal", "dislike", "routine", "fact",
        ])
        if st.button(f"🗑️ Revoke all '{trait_type}' memories"):
            r = delete(f"/memory/cluster/{user_id}/{trait_type}")
            st.success(r.json())
            st.rerun()
    except Exception as e:
        st.error(f"Backend error: {e}")
