"""Regression checks use synthetic data and mocked upstream responses only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(DATA_DIR=tempfile.mkdtemp(prefix="memory-recovery-"), EMBEDDER="hash",
                  LLM_EXTRACT="0", RERANKER="0", MEMORY_API_KEY="test-only")

import httpx
from fastapi.testclient import TestClient
from app.config import Settings
from app.llm_extract import LLMExtractionError, llm_facts, parse_fact_payload
import app.main as api


class GatewayRecoveryTests(unittest.TestCase):
    def test_standard_json_is_unchanged(self):
        payload = {"facts": [{"value": "Nanjing", "subject": "user"}]}
        self.assertEqual(parse_fact_payload(json.dumps(payload)), payload)

    def test_observed_gateway_wrapper_preserves_strings(self):
        payload = {"facts": [{"value": r'C:\notes\[keep] "quoted"', "subject": "user"}]}
        text = "<escape" + json.dumps(payload).replace('"facts": [', '"facts": \\[')
        text = text[:-2] + "\\]}"
        self.assertEqual(parse_fact_payload(text), payload)

    def test_fenced_json(self):
        self.assertEqual(parse_fact_payload('```json\n{"facts": []}\n```'), {"facts": []})

    def test_observed_prose_and_envelope_ending(self):
        self.assertEqual(parse_fact_payload('```\nSynthetic explanation\n```\n<escape{"facts": \\[\\]}\\</'),
                         {"facts": []})
        with self.assertRaises(LLMExtractionError):
            parse_fact_payload('<escape{"facts": \\[\\]} arbitrary trailing text')

    def test_observed_bracket_escaping_without_envelope(self):
        self.assertEqual(parse_fact_payload('{"facts": \\[\\]}'), {"facts": []})

    def test_marker_in_normal_json_string_is_not_an_envelope(self):
        payload = {"facts": [{"value": "literal <escape text"}]}
        self.assertEqual(parse_fact_payload(json.dumps(payload)), payload)

    def test_malformed_and_unsupported_shapes_are_rejected(self):
        for text in ['broken', '{"facts": "not a list"}', '[]',
                     'prose {"facts": []}', '<escape{"facts": \\[{"value": "x"}']:
            with self.subTest(text=text), self.assertRaises(LLMExtractionError):
                parse_fact_payload(text)

    def test_strict_extraction_surfaces_invalid_json(self):
        s = Settings(llm_extract=True, llm_extract_strict=True, openai_api_key="test-only")
        reply = httpx.Response(200, json={"choices": [{"message": {"content": "broken"}}]},
                               request=httpx.Request("POST", "https://example.invalid/chat/completions"))
        with patch('app.llm_extract.httpx.post', return_value=reply), self.assertRaises(LLMExtractionError):
            llm_facts(s, "Synthetic memory")

    def test_truncated_response_is_rejected(self):
        s = Settings(llm_extract=True, llm_extract_strict=True, openai_api_key="test-only")
        reply = httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": '{"facts": []}'}}]},
                               request=httpx.Request("POST", "https://example.invalid/chat/completions"))
        with patch('app.llm_extract.httpx.post', return_value=reply), self.assertRaises(LLMExtractionError):
            llm_facts(s, "Synthetic memory")

    def test_failed_extraction_does_not_commit_and_can_retry(self):
        payload = {"request_id": "recovery-failure", "user_id": "test-recovery", "session_id": "s",
                   "messages": [{"role": "user", "content": "I currently live in Nanjing."}]}
        client = TestClient(api.app)
        headers = {"Authorization": "Bearer test-only"}
        with patch.object(api.settings, 'llm_extract', True):
            with patch('app.main.llm_facts', side_effect=LLMExtractionError('synthetic failure')):
                response = client.post('/add', headers=headers, json=payload)
                self.assertEqual(response.status_code, 503)
                self.assertFalse(api.store.add_completed(payload['request_id'], payload['user_id'], 's'))
            with patch('app.main.llm_facts', return_value=[]) as upstream:
                self.assertEqual(client.post('/add', headers=headers, json=payload).status_code, 200)
                self.assertEqual(upstream.call_count, 1)
            with patch('app.main.llm_facts', side_effect=AssertionError('replay must not call upstream')):
                self.assertEqual(client.post('/add', headers=headers, json=payload).status_code, 200)
                conflict = dict(payload, user_id='different-user')
                self.assertEqual(client.post('/add', headers=headers, json=conflict).status_code, 409)


if __name__ == '__main__':
    unittest.main()
