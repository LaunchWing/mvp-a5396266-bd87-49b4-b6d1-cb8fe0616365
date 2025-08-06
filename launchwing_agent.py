from __future__ import annotations

import os
import subprocess
import json
from typing import List, AsyncGenerator

from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError
import openai
import asyncio

from agents import Agent, ItemHelpers, Runner, function_tool

# -----------------------------------------------------------------------------
# FileSpec data model

class FileSpec(BaseModel):
    path: str
    content: str  # In planning mode, content = purpose

# -----------------------------------------------------------------------------
# Tools

@function_tool()
def create_project(files: List[FileSpec]) -> str:
    created = 0
    for file in files:
        dirname = os.path.dirname(file.path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(file.path, "w", encoding="utf-8") as f:
            f.write(file.content)
        created += 1
    return f"Created {created} file(s)."

@function_tool()
def generate_code(filename: str, description: str) -> str:
    MAX_DESC_LENGTH = 4000
    if not description or len(description.strip()) == 0:
        return f"❌ Empty description for {filename}"

    if len(description) > MAX_DESC_LENGTH:
        description = description[:MAX_DESC_LENGTH]
        print(f"⚠️ Truncated description for {filename} to {MAX_DESC_LENGTH} characters")

    print(f"🛠️ Generating: {filename} (desc length: {len(description)})")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable must be set.")

    client = openai.OpenAI(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are LaunchWing's AI code generator.\n\n"
                "You will be given a file name and a high-level description.\n"
                "Your job is to generate ONLY the full, valid code for that file.\n"
                "Do not include any explanations, markdown, or commentary.\n\n"
                "If the file is a backend handler, it must contain exactly one export:\n"
                "- either `export async function onRequest()`\n"
                "- or `export default { fetch() { ... } }`\n\n"
                "If the file is a config file (e.g. wrangler.toml), output full contents."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Filename: {filename}\n\n"
                f"Description:\n{description}\n\n"
                "Write ONLY the full contents of this file."
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
            timeout=30
        )
    except Exception as e:
        return f"❌ OpenAI call failed for {filename}: {str(e)}"

    code = response.choices[0].message.content or ""
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    return f"✅ Generated {filename}"

@function_tool()
def run_command(command: str) -> str:
    allowed_prefixes = ["pip ", "pytest", "python "]
    if not any(command.startswith(prefix) for prefix in allowed_prefixes):
        return f"Command not permitted: {command}"

    completed = subprocess.run(
        command, shell=True, capture_output=True, text=True, encoding="utf-8"
    )
    output = completed.stdout + completed.stderr
    return output.strip()

@function_tool()
def save_results(message: str) -> str:
    with open("build_log.txt", "a", encoding="utf-8") as log:
        log.write(message + "\n")
    return "Log entry saved."

# -----------------------------------------------------------------------------
# Planning Step

def plan_project(requirements: str) -> List[FileSpec]:
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    system = (
        "You are an expert project planner. Given the user's requirements, return a JSON array "
        "of files to create. Each object should have a `path` and a `purpose`. Keep the list short, focused, and useful."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Requirements: {requirements}\n\nReturn a list of files this app should include, and why."}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )

    content = response.choices[0].message.content or ""
    print("🧠 File plan:\n", content)

    try:
        parsed = json.loads(content)
        return [FileSpec(path=f["path"], content=f["purpose"]) for f in parsed]
    except Exception as e:
        print("❌ Failed to parse file plan:", e)
        return []

# -----------------------------------------------------------------------------
# Agent

AGENT_INSTRUCTIONS = (
    "You are LaunchWing's AppBuilder agent. You build software projects from"
    " natural‑language requirements using the provided tools.\n"
    "Follow a clear plan. Use generate_code for each file."
)

agent = Agent(
    name="LaunchWingAppBuilder",
    instructions=AGENT_INSTRUCTIONS,
    tools=[create_project, generate_code, run_command, save_results],
    model="gpt-4o",
)

# -----------------------------------------------------------------------------
# High-Level Agent Runners

def build_app(requirements: str) -> str:
    run = Runner.run_sync(agent, requirements)
    return run.final_output

async def stream_build_app(requirements: str) -> AsyncGenerator[str, None]:
    result = Runner.run_streamed(agent, input=requirements)
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            continue
        if event.type == "agent_updated_stream_event":
            yield f"Agent updated: {event.new_agent.name}"
            continue
        if event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item":
                try:
                    yield f"Tool call: {item.tool}"
                except Exception:
                    pass
            elif item.type == "tool_call_output_item":
                try:
                    yield f"Tool output from {item.tool}: {item.output}"
                except Exception:
                    pass
            elif item.type == "message_output_item":
                yield ItemHelpers.text_message_output(item)
    if result.final_output:
        yield result.final_output

# -----------------------------------------------------------------------------
# Return Generated Files as JSON (with file planning)

def build_app_files(requirements: str) -> List[FileSpec]:
    plan = plan_project(requirements)

    for file in plan:
        description = f"User requirements:\n{requirements}\n\nGenerate the full file: {file.path}\nPurpose: {file.content}"
        generate_code(file.path, description)

    collected_files = []
    MAX_FILES = 40
    MAX_FILE_SIZE = 200_000

    for dirpath, _, filenames in os.walk("."):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, ".")

            if not rel_path.endswith((".html", ".ts", ".js", ".css", ".json", ".txt", ".md", ".py", ".toml", ".yml")):
                continue

            try:
                size = os.path.getsize(full_path)
                if size > MAX_FILE_SIZE:
                    print(f"⚠️ Skipping oversized file: {rel_path} ({size} bytes)")
                    continue

                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                collected_files.append(FileSpec(path=rel_path, content=content))
                print(f"✅ Included file: {rel_path} ({len(content)} bytes)")

                if len(collected_files) >= MAX_FILES:
                    print(f"⛔ File limit reached: {MAX_FILES}")
                    break
            except Exception as e:
                print(f"❌ Failed to read file {rel_path}: {e}")
                continue

        if len(collected_files) >= MAX_FILES:
            break

    return collected_files