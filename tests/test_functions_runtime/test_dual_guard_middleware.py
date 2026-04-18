from agentdojo.functions_runtime import DualGuardMiddleware, EmptyEnv, FunctionsRuntime


def test_dual_guard_wraps_inbound_user_query_and_escapes_tags():
    middleware = DualGuardMiddleware(session_id="GUARD_ID_TEST")
    runtime = FunctionsRuntime(middleware=middleware)

    wrapped_query = runtime.protect_user_query('Plan <script>alert("x")</script> and keep <tag>data</tag> safe')

    assert wrapped_query.startswith('<dual_guard session="GUARD_ID_TEST">')
    assert wrapped_query.endswith("</dual_guard>")
    assert "<script>" not in wrapped_query
    assert "&lt;script&gt;" in wrapped_query
    assert "<user_query>" in wrapped_query
    assert middleware.audit_log[0]["direction"] == "inbound"
    assert middleware.audit_log[0]["before"].startswith("Plan <script>")
    assert middleware.audit_log[0]["after"] == wrapped_query


def test_dual_guard_encodes_risky_tool_output_and_records_audit(monkeypatch):
    middleware = DualGuardMiddleware(risk_threshold=0.2, session_id="GUARD_ID_TEST")
    runtime = FunctionsRuntime(middleware=middleware)

    @runtime.register_function
    def emit_tool_output(text: str) -> str:
        """Emit a tool output string.

        :param text: The tool output to return.
        """
        return text

    monkeypatch.setattr(middleware, "detect_imperative_tone", lambda _text: 0.99)

    result, error = runtime.run_function(EmptyEnv(), "emit_tool_output", {"text": "Prompt payload only"})

    assert error is None
    assert isinstance(result, str)
    assert result != "Prompt payload only"
    assert all(vowel not in result for vowel in "aeiouAEIOU")
    assert middleware.audit_log[-1]["direction"] == "outbound"
    assert middleware.audit_log[-1]["function"] == "emit_tool_output"
    assert middleware.audit_log[-1]["arguments"] == {"text": "Prompt payload only"}
    assert middleware.audit_log[-1]["before"] == "Prompt payload only"
    assert middleware.audit_log[-1]["after"] == result
