import streamlit as st
import traceback
from agent.graph import agent

st.set_page_config(page_title="Project Planner")

st.title("Engineering Project Planner")

user_prompt = st.text_area("Enter your project prompt:")

if st.button("Run"):
    try:
        if not user_prompt.strip():
            st.warning("Please enter a prompt")
        else:
            result = agent.invoke(
                {"user_prompt": user_prompt},
                {"recursion_limit": 100}
            )
            st.success("Result generated")
            st.write(result)

    except Exception:
        st.error("Error occurred")
        st.text(traceback.format_exc())