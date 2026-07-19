import os
import re
import time
import uuid
import json
import logging
from typing import Optional, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("npmai_mcp")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

_supabase = None


def db():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError(
                "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY must be set as env vars "
                "on this server. Never ship the service-role key to the desktop app."
            )
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _supabase


def get_user_id_for_token(token: str) -> Optional[str]:
    res = db().table("mcp_links").select("user_id").eq("token", token).limit(1).execute()
    if res.data:
        return res.data[0]["user_id"]
    return None


_DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/(?:\s|$)",
    r"shutil\.rmtree\(\s*['\"]?/(?:\s|['\"])",
    r"os\.system\(\s*['\"]rm\s+-rf",
    r"format\s+[cC]:",
    r"del\s+/[sS]\s+/[qQ]\s+[cC]:\\",
    r":\(\)\{.*\|.*&\};:",  
    r"DROP\s+DATABASE",
    r"DROP\s+TABLE",
]


def looks_dangerous(code: str) -> Optional[str]:
    for pat in _DANGEROUS_PATTERNS:
        if re.search(pat, code, re.IGNORECASE):
            return pat
    return None


#Prompt Builders

"""These prompts which is here is not final we will review it again and update it so
although this is not yet deployed from our side and if you use then take care of the prompts that you need to change a lot 
of things here and this is not currently you can use as it is"""

def planner_prompt(task_text: str) -> str:
    return (
        "You are the Planner. Break the following task into 2-6 short, ordered, "
        "concrete steps. Respond with ONLY a JSON object: "
        '{"summary": "...", "steps": ["step 1", "step 2", ...]}\n\n'
        f"Task: {task_text}"
    )


def tool_manager_prompt(plan_json: str) -> str:
    return (
        "You are the Tool Manager. Given this plan, decide what each step needs "
        "(local file/OS access, a specific external API, or nothing beyond your "
        "own reasoning). Respond with ONLY a JSON object mapping step index to a "
        "short description of what's needed: "
        '{"0": "needs local filesystem access to rename files", "1": "..."}\n\n'
        f"Plan: {plan_json}"
    )


def coder_prompt(plan_json: str, tools_json: str) -> str:
    return (
        "You are the Coder. Write a single self-contained Python script that "
        "performs the ENTIRE plan below, step by step, printing progress as it "
        "goes. It will run unmodified as a subprocess on the user's own machine. "
        "Respond with ONLY the raw Python code, no markdown fences, no explanation.\n\n"
        f"Plan: {plan_json}\nTool needs per step: {tools_json}"
    )


def auditor_prompt(code: str) -> str:
    return (
        "You are the Auditor. Review the Python code below for anything "
        "destructive, irreversible, or outside the plan's stated scope "
        "(mass deletion, formatting drives, exfiltrating credentials, network "
        "calls to unexpected hosts, etc). Respond with ONLY one line: either "
        '"ALLOW" if it is safe to run as-is, or "BLOCK: <short reason>" if not.\n\n'
        f"Code:\n{code}"
    )


def verifier_prompt(code: str, output: str) -> str:
    return (
        "You are the Verifier. The code below was executed on the user's "
        "machine and produced the output shown. Confirm whether the task "
        "actually completed successfully (not just whether Python exited "
        'without a traceback). Respond with ONLY one line: "VERIFIED: <what '
        'was accomplished>" or "FAILED: <what went wrong>".\n\n'
        f"Code:\n{code}\n\nOutput:\n{output}"
    )


def create_task(user_id: str, task_text: str) -> dict:
    row = {
        "user_id": user_id,
        "stage": "PLANNING",
        "task_text": task_text,
    }
    res = db().table("mcp_tasks").insert(row).execute()
    return res.data[0]


def get_task(user_id: str, task_id: str) -> Optional[dict]:
    res = (
        db().table("mcp_tasks").select("*")
        .eq("id", task_id).eq("user_id", user_id)
        .limit(1).execute()
    )
    return res.data[0] if res.data else None


def update_task(task_id: str, **fields):
    fields["updated_at"] = "now()"
    db().table("mcp_tasks").update(fields).eq("id", task_id).execute()


def enqueue_job(user_id: str, code: str) -> str:
    res = db().table("mcp_jobs").insert({"user_id": user_id, "code": code}).execute()
    return res.data[0]["id"]


def check_job_result(job_id: str) -> Optional[dict]:
    res = (
        db().table("mcp_job_results").select("*")
        .eq("job_id", job_id).limit(1).execute()
    )
    return res.data[0] if res.data else None


def advance(user_id: str, task_id: Optional[str], user_input: str) -> dict:
    """
    Runs one step of the state machine and returns the payload the tool
    result should carry back to the connected LLM.
    """
    if not task_id:
        task = create_task(user_id, user_input)
        return {
            "task_id": task["id"],
            "stage": "PLANNING",
            "done": False,
            "instructions": planner_prompt(user_input),
        }

    task = get_task(user_id, task_id)
    if task is None:
        new_task = create_task(user_id, user_input)
        return {
            "task_id": new_task["id"],
            "stage": "PLANNING",
            "done": False,
            "instructions": (
                "(Note: previous task_id was not found, starting a new task.) "
                + planner_prompt(user_input)
            ),
        }

    stage = task["stage"]

    if stage == "PLANNING":
        update_task(task_id, stage="TOOL_SELECTION", plan=user_input)
        return {"task_id": task_id, "stage": "TOOL_SELECTION", "done": False,
                "instructions": tool_manager_prompt(user_input)}

    if stage == "TOOL_SELECTION":
        update_task(task_id, stage="CODING", selected_tools=user_input)
        return {"task_id": task_id, "stage": "CODING", "done": False,
                "instructions": coder_prompt(task["plan"], user_input)}

    if stage == "CODING":
        update_task(task_id, stage="AUDITING", code=user_input)
        return {"task_id": task_id, "stage": "AUDITING", "done": False,
                "instructions": auditor_prompt(user_input)}

    if stage == "AUDITING":
        verdict = user_input.strip()
        update_task(task_id, audit_result=verdict)

        if verdict.upper().startswith("BLOCK"):
            update_task(task_id, stage="FAILED")
            return {"task_id": task_id, "stage": "FAILED", "done": True,
                    "instructions": f"Task blocked at audit stage: {verdict}"}

        hit = looks_dangerous(task["code"])
        if hit:
            update_task(task_id, stage="FAILED")
            return {"task_id": task_id, "stage": "FAILED", "done": True,
                    "instructions": f"Task blocked by safety pre-check (pattern: {hit}), "
                                    f"regardless of audit verdict."}

        job_id = enqueue_job(user_id, task["code"])
        update_task(task_id, stage="EXECUTING", job_id=job_id)
        return {"task_id": task_id, "stage": "EXECUTING", "done": False,
                "instructions": (
                    "Code passed audit and was sent to the user's desktop app for "
                    "execution. Call this tool again with the same task_id and any "
                    "input (e.g. 'checking') to poll for the result."
                )}

    if stage == "EXECUTING":
        result = check_job_result(task["job_id"])
        if result is None:
            return {"task_id": task_id, "stage": "EXECUTING", "done": False,
                    "instructions": (
                        "Still executing on the user's desktop — call this tool again "
                        "with the same task_id in a few seconds."
                    )}
        update_task(task_id, stage="VERIFYING", result_output=result["output"])
        return {"task_id": task_id, "stage": "VERIFYING", "done": False,
                "instructions": verifier_prompt(task["code"], result["output"])}

    if stage == "VERIFYING":
        update_task(task_id, stage="DONE", verify_result=user_input)
        return {"task_id": task_id, "stage": "DONE", "done": True,
                "instructions": f"Task complete. {user_input}"}

    return {"task_id": task_id, "stage": stage, "done": True,
            "instructions": "This task has already finished. Start a new task with an "
                             "empty task_id if you have something else to do."}


TOOL_SCHEMA = {
    "name": "npmai_agent",
    "description": (
        "Runs a task on the user's own computer through the NPMAI Agent pipeline "
        "(plan -> select tools -> generate code -> audit -> execute on the user's "
        "desktop -> verify). Call it once with just `input` set to the task "
        "description to start. Every following call MUST include the `task_id` "
        "from the previous response, with `input` set to exactly what the "
        "previous response's `instructions` field asked you to produce. Keep "
        "calling until the response's `done` field is true."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Leave empty/omit to start a new task. Otherwise, "
                                "the task_id from the previous call's result.",
            },
            "input": {
                "type": "string",
                "description": "New task: the task description. Continuing: your "
                                "response to the previous `instructions` field.",
            },
        },
        "required": ["input"],
    },
}


def call_tool(user_id: str, arguments: dict) -> dict:
    task_id = arguments.get("task_id") or None
    user_input = arguments.get("input", "")
    return advance(user_id, task_id, user_input)

def jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_rpc(body: dict, user_id: str) -> Optional[dict]:
    method = body.get("method")
    req_id = body.get("id")
    params = body.get("params") or {}
    is_notification = "id" not in body

    if method == "initialize":
        client_protocol = params.get("protocolVersion", "2025-06-18")
        return jsonrpc_result(req_id, {
            "protocolVersion": client_protocol,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "npmai-agent-mcp", "version": "1.0.0"},
        })

    if method == "notifications/initialized":
        return None 

    if method == "ping":
        return jsonrpc_result(req_id, {})

    if method == "tools/list":
        return jsonrpc_result(req_id, {"tools": [TOOL_SCHEMA]})

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if tool_name != TOOL_SCHEMA["name"]:
            return jsonrpc_error(req_id, -32602, f"Unknown tool: {tool_name}")
        try:
            payload = call_tool(user_id, arguments)
        except Exception as e:
            log.exception("tool call failed")
            return jsonrpc_result(req_id, {
                "isError": True,
                "content": [{"type": "text", "text": f"Internal error: {e}"}],
            })
        return jsonrpc_result(req_id, {
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "isError": False,
        })

    if is_notification:
        return None
    return jsonrpc_error(req_id, -32601, f"Method not found: {method}")


app = FastAPI(title="NPMAI Agent MCP Server")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/mcp/{token}")
async def mcp_endpoint(token: str, request: Request):
    user_id = get_user_id_for_token(token)
    if user_id is None:
        return JSONResponse({"error": "unknown or revoked link"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            jsonrpc_error(None, -32700, "Parse error: invalid JSON"), status_code=400
        )

    result = handle_rpc(body, user_id)

    if result is None:
        return Response(status_code=202)

    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        sse_body = f"event: message\ndata: {json.dumps(result)}\n\n"
        return Response(content=sse_body, media_type="text/event-stream")

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
  """ Above port 7860 is we used as we deploy on huggingface so if you are uisng this code remember to change the port or at
  least review your environment."""
