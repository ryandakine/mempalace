# mempalace — agent guide

ChromaDB-backed memory store + MCP server. The on-disk palace format is tied to the
chromadb 0.6.x line; `mempalace/config.py` guards every palace open
(`open_palace_client` / `get_palace_collection`) and raises `UnsupportedChromaDBError`
for any other chromadb. Never open `~/.mempalace/palace` with another interpreter's
chromadb. Tests: `.venv/bin/python -m pytest -q`. Lint: `.venv/bin/python -m ruff check mempalace tests`.

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
