"""Tool categories shared by trace analysis and the replay runtime."""

CATEGORIES = {
    "shell": {"exec_command", "Bash", "shell", "shell_command", "write_stdin",
              "BashOutput", "KillShell", "local_shell"},
    "read": {"Read", "Grep", "Glob", "LS", "view_image", "NotebookRead"},
    "edit": {"Edit", "Write", "MultiEdit", "apply_patch", "NotebookEdit"},
    "web": {"WebFetch", "WebSearch", "web_search"},
    "subagent": {"Agent", "Task", "TaskOutput", "wait_agent", "spawn_agent"},
    # Time a human spends answering; not machine work, excluded from T.
    "human": {"AskUserQuestion", "request_user_input"},
    "plan": {"TaskUpdate", "TaskCreate", "TaskList", "TaskGet", "TodoWrite",
             "update_plan", "ExitPlanMode", "EnterPlanMode"},
}
_LOOKUP = {name: cat for cat, names in CATEGORIES.items() for name in names}


def category(tool_name: str | None) -> str:
    if tool_name is None:
        return "other"
    if tool_name.startswith("mcp__"):
        return "mcp"
    return _LOOKUP.get(tool_name, "other")
