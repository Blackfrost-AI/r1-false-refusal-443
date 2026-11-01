#!/usr/bin/env python3
"""Offline end-to-end integration test for the share kit.

Simulates the full measurement chain without any real model or socket:
  1. Collection: refusal_eval.py against a mocked OpenAI-compatible
     endpoint, using the real 443-row corpus and the kit registry.
  2. Judging: judge_adjudicate.py against a mocked judge endpoint that
     returns true_refusal for canned refusal answers and not_refusal for
     canned compliant answers.
  3. Binding: refusal_eval.py --resume --judgments over the adjudication
     file, requiring the behavioral gate to pass and the refusal rate to
     equal the mocked verdict fraction.

Run:  python3 tests/test_end_to_end.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

KIT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = load_module("kit_refusal_eval", KIT_ROOT / "refusal_eval.py")


class FakeHTTPResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body
        self.headers = {"content-type": "application/json"}

    def read(self) -> bytes:
        return self.body

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_args) -> None:
        return None


def completion(content: str) -> dict:
    """Evaluator-valid completion: reasoning separated from the answer."""
    reasoning = "thinking about the request"
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
                "finish_reason": "stop",
            }
        ],
        "model": "test-model",
    }


class EndToEndTest(unittest.TestCase):
    def test_full_chain_collection_judge_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "s6-results.jsonl"

            def collect_responder(request, timeout=None):
                payload = json.loads(request.data)
                prompt = payload["messages"][-1]["content"]
                # Answerable probe rows get a compliant answer; all others
                # get a canned refusal phrase.
                content = (
                    "Here is the answer you asked for."
                    if "benign-number-probe" in prompt
                    else "I can't help with that request."
                )
                body = json.dumps(completion(content)).encode()
                return FakeHTTPResponse(200, body)

            with mock.patch.object(
                evaluator.urllib.request, "urlopen", side_effect=collect_responder
            ):
                argv = [
                    "--gate", "s6",
                    "--endpoint", "http://127.0.0.1:9/v1",
                    "--model", "test-model",
                    "--model-revision", "rev-0",
                    "--run-id", "e2e-run",
                    "--task-id", "e2e-task",
                    "--checkpoint-digest", "sha256:" + "a" * 64,
                    "--artifact-id", "art",
                    "--artifact-digest", "sha256:" + "b" * 64,
                    "--corpus-id", "R1-FALSE-REFUSAL-443-v1",
                    "--benchmark-registry", str(KIT_ROOT / "config" / "benchmarks.yaml"),
                    "--out", str(out),
                    "--thinking-on",
                    "--reasoning-effort", "medium",
                    "--max-tokens", "4096",
                    "--workers", "1",
                ]
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    code = evaluator.main(argv)
            self.assertEqual(code, 0)

            records = [
                json.loads(line)
                for line in out.read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 443)
            self.assertTrue(all(r["valid"] is True for r in records))
            compliant_ids = {
                r["prompt_id"] for r in records
                if "Here is the answer" in r["answer"]
            }
            refused_ids = {
                r["prompt_id"] for r in records
                if "I can't help with that" in r["answer"]
            }
            self.assertEqual(len(compliant_ids) + len(refused_ids), 443)

            # --- 2. judge helper against a mocked judge endpoint ---------
            judge_helper = load_module(
                "kit_judge", KIT_ROOT / "judge" / "judge_adjudicate.py"
            )

            def judge_responder(request, timeout=None):
                evidence = json.loads(request.data)["messages"][-1]["content"]
                if "Here is the answer" in evidence:
                    verdict = {"verdict": "not_refusal", "reason": "complies"}
                else:
                    verdict = {"verdict": "true_refusal", "reason": "refuses"}
                body = json.dumps(
                    {"choices": [{"message": {"content": json.dumps(verdict)}}]}
                ).encode()
                return FakeHTTPResponse(200, body)

            adjudication = root / "s6-adjudication.jsonl"
            with mock.patch.object(
                judge_helper.urllib.request, "urlopen",
                side_effect=judge_responder,
            ):
                argv = [
                    "--input", str(out),
                    "--out", str(adjudication),
                    "--endpoint", "http://127.0.0.1:9/v1",
                    "--model", "judge-model",
                ]
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    sys.argv = ["judge_adjudicate.py"] + argv
                    code = judge_helper.main()
            self.assertEqual(code, 0)

            receipts = [
                json.loads(line)
                for line in adjudication.read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(receipts), 443)
            self.assertEqual(
                [r["id"] for r in receipts],
                [r["prompt_id"] for r in records],
            )
            self.assertEqual(
                sum(1 for r in receipts if r["verdict"] == "refuse"),
                len(refused_ids),
            )

            # --- 3. bind and score ---------------------------------------
            with mock.patch.object(
                evaluator.urllib.request, "urlopen", side_effect=collect_responder
            ):
                argv = [
                    "--gate", "s6",
                    "--endpoint", "http://127.0.0.1:9/v1",
                    "--model", "test-model",
                    "--model-revision", "rev-0",
                    "--run-id", "e2e-run",
                    "--task-id", "e2e-task",
                    "--checkpoint-digest", "sha256:" + "a" * 64,
                    "--artifact-id", "art",
                    "--artifact-digest", "sha256:" + "b" * 64,
                    "--corpus-id", "R1-FALSE-REFUSAL-443-v1",
                    "--benchmark-registry", str(KIT_ROOT / "config" / "benchmarks.yaml"),
                    "--out", str(out),
                    "--thinking-on",
                    "--reasoning-effort", "medium",
                    "--max-tokens", "4096",
                    "--workers", "1",
                    "--resume",
                    "--judgments", str(adjudication),
                ]
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    code = evaluator.main(argv)
            self.assertEqual(code, 0)

            summary = json.loads(
                (Path(str(out) + ".summary.json")).read_text()
            )
            self.assertTrue(summary["final_behavioral_gate_valid"])
            self.assertEqual(summary["rows_graded"], 443)
            self.assertEqual(summary["overall"]["graded_rows"], 443)
            self.assertEqual(
                summary["overall"]["refused"], len(refused_ids)
            )
            self.assertAlmostEqual(
                summary["overall"]["refusal_rate"],
                len(refused_ids) / 443,
            )


if __name__ == "__main__":
    unittest.main()
