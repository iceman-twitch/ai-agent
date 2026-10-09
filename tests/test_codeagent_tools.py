import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import codeagent


class WorkspaceToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name).resolve()
        self.agent = codeagent.CodeAgent(client=Mock(), workspace=self.workspace)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_reads_and_searches_project_files(self) -> None:
        source = self.workspace / "src" / "example.py"
        source.parent.mkdir()
        source.write_text("print('needle')\nsecond\nthird\n", encoding="utf-8")
        (self.workspace / ".env").write_text("ANTHROPIC_API_KEY=secret", encoding="utf-8")
        (self.workspace / ".git").mkdir()
        (self.workspace / ".git" / "config").write_text("needle", encoding="utf-8")

        content, failed = self.agent.execute_tool("read_file", {"path": "src/example.py"})
        self.assertFalse(failed)
        self.assertEqual(content, "print('needle')\nsecond\nthird\n")

        content, failed = self.agent.execute_tool(
            "read_file",
            {"path": "src/example.py", "start_line": 2, "max_lines": 1},
        )
        self.assertFalse(failed)
        self.assertEqual(content, "2: second")

        results, failed = self.agent.execute_tool("search_files", {"query": "needle"})
        self.assertFalse(failed)
        self.assertIn(f"{Path('src') / 'example.py'}:1:", results)
        self.assertNotIn(".git", results)
        self.assertNotIn(".env", results)

    def test_large_file_reads_are_bounded_and_can_be_chunked(self) -> None:
        large_file = self.workspace / "large.txt"
        large_file.write_text("\n".join(f"line-{line}" for line in range(1, 5001)), encoding="utf-8")

        result, failed = self.agent.execute_tool("read_file", {"path": "large.txt"})

        self.assertFalse(failed)
        self.assertLessEqual(len(result), codeagent.MAX_TOOL_OUTPUT_CHARS + 100)
        self.assertIn("use start_line", result)

        result, failed = self.agent.execute_tool(
            "read_file",
            {"path": "large.txt", "start_line": 4999, "max_lines": 2},
        )
        self.assertFalse(failed)
        self.assertEqual(result, "4999: line-4999\n5000: line-5000")

    def test_blocks_paths_outside_project_and_secrets(self) -> None:
        for path in ("../outside.txt", ".env", ".env.local", ".envrc", ".git/config"):
            with self.subTest(path=path):
                result, failed = self.agent.execute_tool("read_file", {"path": path})
                self.assertTrue(failed)
                self.assertIn("Tool error:", result)

    def test_write_requires_approval(self) -> None:
        with patch.object(self.agent, "_request_approval", return_value=False) as approval:
            result, failed = self.agent.execute_tool(
                "write_file",
                {"path": "new.txt", "content": "not written"},
            )

        self.assertTrue(failed)
        self.assertIn("declined", result)
        self.assertFalse((self.workspace / "new.txt").exists())
        approval.assert_called_once()

    def test_approved_write_and_delete(self) -> None:
        target = self.workspace / "data.txt"
        with patch.object(self.agent, "_request_approval", return_value=True):
            result, failed = self.agent.execute_tool(
                "write_file",
                {"path": "data.txt", "content": "saved"},
            )
        self.assertFalse(failed)
        self.assertIn("Wrote", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "saved")

        with patch.object(self.agent, "_request_approval", return_value=True):
            result, failed = self.agent.execute_tool("delete_file", {"path": "data.txt"})
        self.assertFalse(failed)
        self.assertIn("Deleted", result)
        self.assertFalse(target.exists())

    def test_rename_and_shell_command_require_approval(self) -> None:
        source = self.workspace / "old.txt"
        source.write_text("data", encoding="utf-8")
        with patch.object(self.agent, "_request_approval", return_value=False):
            result, failed = self.agent.execute_tool(
                "rename_file",
                {"source": "old.txt", "destination": "new.txt"},
            )
        self.assertTrue(failed)
        self.assertTrue(source.exists())
        self.assertFalse((self.workspace / "new.txt").exists())

        with (
            patch.object(self.agent, "_request_approval", return_value=False),
            patch("codeagent.subprocess.run") as run_command,
        ):
            result, failed = self.agent.execute_tool(
                "run_command",
                {"command": "git status"},
            )
        self.assertTrue(failed)
        self.assertIn("declined", result)
        run_command.assert_not_called()

    def test_approved_shell_command_runs_from_project_root(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="working tree clean", stderr="")
        with (
            patch.object(self.agent, "_request_approval", return_value=True),
            patch("codeagent.subprocess.run", return_value=completed) as run_command,
        ):
            result, failed = self.agent.execute_tool(
                "run_command",
                {"command": "git status"},
            )

        self.assertFalse(failed)
        self.assertIn("working tree clean", result)
        self.assertEqual(run_command.call_args.kwargs["cwd"], self.workspace)
        self.assertTrue(run_command.call_args.kwargs["shell"])

    def test_history_window_keeps_tool_results_with_their_user_turn(self) -> None:
        self.agent.max_history_messages = 1
        self.agent.history = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": [{"type": "text", "text": "old answer"}]},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t"}]},
        ]

        window = self.agent._windowed_history()

        self.assertEqual(window[0]["content"], "new question")
        self.assertEqual(window[-1]["content"][0]["type"], "tool_result")

    def test_project_selection_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as settings_directory:
            state_file = Path(settings_directory) / "project.json"
            with patch.object(codeagent, "PROJECT_STATE_FILE", state_file):
                selected = codeagent.resolve_active_project(str(self.workspace))
                reopened = codeagent.resolve_active_project()

        self.assertEqual(selected, self.workspace)
        self.assertEqual(reopened, self.workspace)

    def test_switching_project_clears_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as settings_directory:
            with patch.object(
                codeagent,
                "PROJECT_STATE_FILE",
                Path(settings_directory) / "project.json",
            ):
                self.agent.history.append({"role": "user", "content": "previous project"})
                with tempfile.TemporaryDirectory() as next_project:
                    selected = Path(next_project).resolve()
                    self.agent.switch_workspace(selected)

        self.assertEqual(self.agent.workspace, selected)
        self.assertEqual(self.agent.history, [])

    def test_tool_loop_executes_tool_then_returns_final_answer(self) -> None:
        tool_use = SimpleNamespace(
            type="tool_use",
            id="tool-1",
            name="read_file",
            input={"path": "src/example.py"},
            model_dump=lambda **_: {
                "type": "tool_use",
                "id": "tool-1",
                "name": "read_file",
                "input": {"path": "src/example.py"},
            },
        )
        final_text = SimpleNamespace(
            type="text",
            text="The project contains the requested file.",
            model_dump=lambda **_: {
                "type": "text",
                "text": "The project contains the requested file.",
            },
        )
        usage = SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )

        class FakeStream:
            def __init__(self, content: list[object], text: str = "") -> None:
                self.text_stream = [text] if text else []
                self.final = SimpleNamespace(content=content, usage=usage)

            def __enter__(self) -> "FakeStream":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def get_final_message(self) -> SimpleNamespace:
                return self.final

        client = codeagent.anthropic.Anthropic(api_key="test")
        self.agent = codeagent.CodeAgent(client=client, workspace=self.workspace)
        self.agent.show_usage = False
        request_messages = []
        responses = [
            FakeStream([tool_use]),
            FakeStream([final_text], "The project contains the requested file."),
        ]

        def fake_stream(**kwargs: object) -> FakeStream:
            request_messages.append(copy.deepcopy(kwargs["messages"]))
            return responses.pop(0)

        with (
            patch.object(client.messages, "stream", side_effect=fake_stream) as stream_request,
            patch.object(self.agent, "execute_tool", return_value=("file contents", False)),
        ):
            answer = self.agent.ask("Inspect src/example.py")

        self.assertEqual(answer, "The project contains the requested file.")
        self.assertEqual(stream_request.call_count, 2)
        second_request = request_messages[1]
        self.assertEqual(
            stream_request.call_args_list[1].kwargs["tools"],
            codeagent.WORKSPACE_TOOLS,
        )
        self.assertEqual(
            second_request[-1]["content"][0]["type"],
            "tool_result",
        )


if __name__ == "__main__":
    unittest.main()
