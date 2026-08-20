from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def run_protocol_check(directory: str, search_tool: str) -> None:
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        root = Path({str(ROOT)!r})
        sys.path.insert(0, str(root / 'baselines' / {directory!r}))
        import run as runner

        handles = [f'page_{{index:06d}}' for index in range(1, 11)]
        class HandleSpace:
            by_handle = {{handle: object() for handle in handles}}
        class Runtime:
            handle_space = HandleSpace()
            discovered_page_handles = set(handles)
            seen_page_handles = set()
            def validate_evidence_handles(self, values):
                return []

        one = {{
            'action': 'submit_answer',
            'ranked_answers': [{{'page_handle': handles[0], 'evidence_page_handles': [], 'rationale': 'x'}}],
            'overall_rationale': 'x',
        }}
        normalized, error = runner.normalize_submit(one, Runtime())
        assert normalized is None and 'exactly 10' in error

        ten = {{
            'action': 'submit_answer',
            'ranked_answers': [
                {{'page_handle': handle, 'evidence_page_handles': [], 'rationale': 'x'}}
                for handle in handles
            ],
            'overall_rationale': 'x',
        }}
        normalized, error = runner.normalize_submit(ten, Runtime())
        assert normalized is not None and not error
        valid_metrics = runner.score_submission(normalized, handles[0])
        assert valid_metrics['gold_rank'] == 1
        assert valid_metrics['recall_at_1'] == 1.0
        assert valid_metrics['mrr_at_10'] == 1.0

        class ShortRuntime:
            handle_space = HandleSpace()
            discovered_page_handles = set(handles[:5])
            seen_page_handles = set()
            def validate_evidence_handles(self, values):
                return []

        short = {{
            'action': 'submit_answer',
            'ranked_answers': [
                {{'page_handle': handle, 'evidence_page_handles': [], 'rationale': 'x'}}
                for handle in handles[:5]
            ],
            'overall_rationale': 'x',
        }}
        normalized, error = runner.normalize_submit(
            short, ShortRuntime(), allow_fewer_than_ten=True
        )
        assert normalized is not None and not error
        assert len(normalized['ranked_answers']) == 5
        invalid_metrics = runner.score_submission(normalized, handles[0])
        assert invalid_metrics == {{
            'gold_rank': None,
            'recall_at_1': 0.0,
            'recall_at_3': 0.0,
            'recall_at_5': 0.0,
            'recall_at_10': 0.0,
            'mrr_at_10': 0.0,
        }}

        search_only = runner.system_prompt_for_tools(frozenset({{{search_tool!r}}}))
        assert {search_tool!r} in search_only
        assert 'inspect_pages(page_handles)' not in search_only
        assert 'crop_pages(crops)' not in search_only

        full = runner.system_prompt_for_tools()
        assert 'inspect_pages(page_handles)' in full
        assert 'crop_pages(crops)' in full
        assert '(0, 0) at the top-left' in full
        assert '(1, 1) at the bottom-right' in full
        assert 'Items are processed independently' in full
        assert runner.HARNESS_VERSION == 'visdocagentbench_agent_harness_v2'
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_visual_agent_protocol() -> None:
    run_protocol_check("agent_visual", "visual_search")


def test_ocr_text_agent_protocol() -> None:
    run_protocol_check("agent_ocr_text", "text_search")


def run_completed_resume_check(directory: str) -> None:
    code = textwrap.dedent(
        f"""
        import json
        import sys
        import tempfile
        from pathlib import Path

        root = Path({str(ROOT)!r})
        sys.path.insert(0, str(root / 'baselines' / {directory!r}))
        import run as runner

        handles = [f'page_{{index:06d}}' for index in range(1, 12)]
        fingerprint = 'current-run'
        base_row = {{
            'query_id': 'q1',
            'level': '1',
            'topic_id': 'old_topic',
            'status': 'submitted',
            'harness_version': runner.HARNESS_VERSION,
            'episode_fingerprint_sha256': fingerprint,
            'submission': {{
                'ranked_answers': [
                    {{'page_handle': handle, 'evidence_page_handles': [], 'rationale': 'x'}}
                    for handle in handles[:10]
                ],
                'overall_rationale': 'x',
            }},
            'metrics': {{
                'gold_rank': None,
                'recall_at_1': 0.0,
                'recall_at_3': 0.0,
                'recall_at_5': 0.0,
                'recall_at_10': 0.0,
                'mrr_at_10': 0.0,
            }},
        }}

        def load(row):
            with tempfile.TemporaryDirectory() as directory_path:
                path = Path(directory_path) / 'predictions.jsonl'
                path.write_text(json.dumps(row) + '\\n', encoding='utf-8')
                return runner.load_resume_rows(
                    path,
                    {{'q1'}},
                    current_query_rows={{
                        'q1': {{'level': '2', 'topic_id': 'new_topic'}}
                    }},
                    expected_gold_page_handles={{'q1': handles[0]}},
                    valid_page_handles=set(handles),
                    required_harness_version=runner.HARNESS_VERSION,
                    expected_fingerprints={{'q1': fingerprint}},
                )

        resumed = load(base_row)
        assert set(resumed) == {{'q1'}}
        assert resumed['q1']['metrics'] == {{
            'gold_rank': 1,
            'recall_at_1': 1.0,
            'recall_at_3': 1.0,
            'recall_at_5': 1.0,
            'recall_at_10': 1.0,
            'mrr_at_10': 1.0,
        }}
        assert resumed['q1']['level'] == '2'
        assert resumed['q1']['topic_id'] == 'new_topic'
        report = runner.build_metric_report(list(resumed.values()))
        assert report['overall']['recall_at_1'] == 1.0
        assert set(report['by_level']) == {{'2'}}
        assert report['by_level']['2']['count'] == 1
        assert set(report['by_topic']) == {{'new_topic'}}
        assert report['by_topic']['new_topic']['count'] == 1

        duplicate = json.loads(json.dumps(base_row))
        duplicate['submission']['ranked_answers'][-1]['page_handle'] = handles[0]
        assert load(duplicate) == {{}}

        unknown = json.loads(json.dumps(base_row))
        unknown['submission']['ranked_answers'][-1]['page_handle'] = 'page_unknown'
        assert load(unknown) == {{}}
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_visual_completed_resume_revalidates_and_rescores() -> None:
    run_completed_resume_check("agent_visual")


def test_ocr_text_completed_resume_revalidates_and_rescores() -> None:
    run_completed_resume_check("agent_ocr_text")


def run_format_error_step_check(directory: str) -> None:
    code = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        root = Path({str(ROOT)!r})
        sys.path.insert(0, str(root / 'baselines' / {directory!r}))
        import run as runner

        handles = [f'page_{{index:06d}}' for index in range(1, 11)]
        class HandleSpace:
            by_handle = {{handle: object() for handle in handles}}
            def page_id_to_handle(self, page_id):
                return handles[0]

        class Runtime:
            handle_space = HandleSpace()
            trace = []
            discovered_page_handles = set(handles)
            seen_page_handles = set()
            def context_artifact_handles(self, max_images):
                return []
            def artifact_paths(self, values):
                return []
            def public_usage(self):
                return {{}}
            def public_per_call_limits(self):
                return {{}}
            def validate_evidence_handles(self, values):
                return []

        malformed = runner.VLMFormatError(
            request_id='step1',
            raw_response='not json',
            reasoning_content='',
            usage={{}},
            attempts=1,
            request_attempts=[{{'attempt': 1, 'outcome': 'format_error', 'retry_scheduled': False}}],
            error=ValueError('invalid JSON'),
        )
        submission = {{
            'action': 'submit_answer',
            'ranked_answers': [
                {{'page_handle': handle, 'evidence_page_handles': [], 'rationale': 'x'}}
                for handle in handles
            ],
            'overall_rationale': 'x',
            '_raw_response': '{{"action":"submit_answer"}}',
            '_usage': {{}},
            '_attempts': 1,
            '_request_attempts': [{{'attempt': 1, 'outcome': 'accepted', 'retry_scheduled': False}}],
        }}
        values = [malformed, submission]
        prompts = []
        def fake_chat(*args, **kwargs):
            prompts.append(kwargs['text_prompt'])
            value = values.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        runner.chat_completion_json = fake_chat
        result = runner.run_episode(
            query_row={{'query_id': 'q1', 'query': 'query', 'gold_page_id': 'gold'}},
            runtime=Runtime(),
            config=SimpleNamespace(max_tokens=3200),
            max_steps=2,
            allow_repair_turn=True,
            force_final_answer_after_max_steps=False,
        )
        assert result['status'] == 'submitted'
        assert [item['step'] for item in result['model_outputs']] == [1, 2]
        assert result['model_outputs'][0]['event_type'] == 'format_error'
        assert result['request_audit']['logical_planner_turns'] == 2
        assert 'could not be parsed as one JSON object' in prompts[1]
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_visual_format_error_consumes_a_step() -> None:
    run_format_error_step_check("agent_visual")


def test_ocr_text_format_error_consumes_a_step() -> None:
    run_format_error_step_check("agent_ocr_text")
