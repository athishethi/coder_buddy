from dotenv import load_dotenv
import os

from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from agent.prompts import *
from agent.states import *
from agent.tools import write_file, read_file, get_current_directory, list_files

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

# ✅ Model fallback system
MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "llama-3.3-8b-instant",
    "mixtral-8x7b-32768"
]

def get_working_llm():
    for model_name in MODEL_CANDIDATES:
        try:
            llm = ChatGroq(
                api_key=api_key,
                model=model_name,
                temperature=0
            )
            llm.invoke("Hello")  # test
            print(f"✅ Using model: {model_name}")
            return llm
        except Exception:
            print(f"❌ Model failed: {model_name}")

    raise ValueError("No working Groq models available")

llm = get_working_llm()

# ------------------ PLANNER ------------------
def planner_agent(state: dict) -> dict:
    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(state["user_prompt"])
    )
    return {"plan": resp}


# ------------------ ARCHITECT ------------------
def architect_agent(state: dict) -> dict:
    plan: Plan = state["plan"]

    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json())
    )

    resp.plan = plan
    return {"task_plan": resp}


# ------------------ CODER ------------------
def coder_agent(state: dict) -> dict:
    coder_state: CoderState = state.get("coder_state")

    if coder_state is None:
        coder_state = CoderState(
            task_plan=state["task_plan"],
            current_step_idx=0
        )

    steps = coder_state.task_plan.implementation_steps

    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    task = steps[coder_state.current_step_idx]

    existing_content = read_file.run(task.filepath)

    system_prompt = coder_system_prompt()
    user_prompt = f"""
Task: {task.task_description}
File: {task.filepath}

Existing content:
{existing_content}

Use write_file(path, content) to save changes.
"""

    tools = [read_file, write_file, list_files, get_current_directory]

    react_agent = create_react_agent(llm, tools)

    react_agent.invoke({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    })

    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}


# ------------------ GRAPH ------------------
graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

graph.add_conditional_edges(
    "coder",
    lambda s: "END" if s.get("status") == "DONE" else "coder",
    {"END": END, "coder": "coder"}
)

graph.set_entry_point("planner")

# ✅ THIS LINE WAS MISSING / BROKEN
agent = graph.compile()

def coder_agent(state: dict) -> dict:
    coder_state: CoderState = state.get("coder_state")

    if coder_state is None:
        coder_state = CoderState(
            task_plan=state["task_plan"],
            current_step_idx=0
        )

    steps = coder_state.task_plan.implementation_steps

    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    task = steps[coder_state.current_step_idx]

    existing_content = read_file.invoke({"path": task.filepath})

    system_prompt = coder_system_prompt()

    user_prompt = f"""
Task: {task.task_description}
File: {task.filepath}

Existing content:
{existing_content}

IMPORTANT:
- Use write_file tool ONLY
- Always provide BOTH:
  path (string)
  content (string)
"""

    tools = [read_file, write_file, list_files, get_current_directory]

    react_agent = create_react_agent(llm, tools)

    try:
        react_agent.invoke({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        })
    except Exception as e:
        print("⚠️ Tool execution failed:", e)

    coder_state.current_step_idx += 1

    return {"coder_state": coder_state}
