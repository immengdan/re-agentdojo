from __future__ import annotations

import dataclasses
import html
import inspect
import logging
import re
import secrets
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Annotated, Any, Concatenate, Generic, TypeAlias, get_origin, get_type_hints, overload

from docstring_parser import parse
from docstring_parser.common import DocstringParam
from pydantic import BaseModel, Field, ValidationError, create_model
from typing_extensions import ParamSpec, TypeVar


class TaskEnvironment(BaseModel):
    """Base class for the environment in which the agent is running."""

    ...


class EmptyEnv(TaskEnvironment):
    """An empty environment for when a state is not needed."""

    ...


@dataclasses.dataclass(frozen=True)
class Depends:
    """A dependency for a function. It is used to extract information from the environment to pass to the function."""

    env_dependency: str | Callable[[BaseModel], BaseModel]
    """The name of the attribute to extract from the environment or a function that returns the attribute from the environment."""

    def extract_dep_from_env(self, env: BaseModel) -> BaseModel:
        """Extracts the dependency from the environment."""
        if callable(self.env_dependency):
            return self.env_dependency(env)
        return getattr(env, self.env_dependency)

    def __repr__(self) -> str:
        return f"Depends('{self.env_dependency}')"


class FunctionCall(BaseModel):
    """An object containing information about a function call requested by an agent."""

    function: str
    """The name of the function to call."""
    args: MutableMapping[str, "FunctionCallArgTypes"]
    """The arguments to pass to the function."""
    id: str | None = None
    """An optional ID for the function call. E.g., used by OpenAI and Anthropic."""
    placeholder_args: Mapping[str, "FunctionCallArgTypes"] | None = None
    """An optional dictionary of placeholder arguments to use in by ground truth agent in injection tasks."""


FunctionCallArgTypes = str | int | float | bool | None | dict | list | FunctionCall
"""Valid types for function call arguments."""


FunctionReturnType: TypeAlias = BaseModel | Sequence["FunctionReturnType"] | dict | str | int | float | bool | None
"""Union of valid return types for functions. The default [FunctionsRuntime][agentdojo.functions_runtime.FunctionsRuntime]
is not guaranteed to work with other types."""
P = ParamSpec("P")
S = TypeVar("S")
Env = TypeVar("Env", bound=TaskEnvironment)
T = TypeVar("T", bound=FunctionReturnType)
FunctionCallable = Callable[Concatenate[Env, P], FunctionReturnType]


class Function(BaseModel, Generic[P, T]):
    """A function that can be called by an agent and run by a [`FunctionsRuntime`][agentdojo.functions_runtime.FunctionsRuntime]."""

    name: str
    """The name of the function."""
    description: str
    """A short description of the function."""
    parameters: type[BaseModel]
    """The input schema for the function. It is defined as a Pydantic model."""
    dependencies: dict[str, Depends]
    """A dictionary of dependencies for the function. See [`Depends`][agentdojo.functions_runtime.Depends] for more information."""
    run: Callable[P, T]
    """The function implementation."""
    full_docstring: str
    """The full docstring of the functions."""
    return_type: Any | None
    """The return type of the function."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.run(*args, **kwargs)


def _get_dependencies(function: Callable[P, T]) -> dict[str, Depends]:
    arg_types = get_type_hints(function, include_extras=True)
    dependencies: dict[str, Depends] = {}
    for arg_name, arg_type in arg_types.items():
        if get_origin(arg_type) is Annotated:
            if isinstance(dependency := arg_type.__metadata__[0], Depends):
                dependencies[arg_name] = dependency
    return dependencies


def make_function(function: Callable[P, T]) -> Function:
    """Creates a [`Function`][agentdojo.functions_runtime.Function] instance from a function.

    It does so by extracting the docstrings and type hints from the function. It then creates
    a Pydantic model for the input schema of the function.

    Note:
        ReStructuredText is the recommended docstring format.

    Args:
        function: The function to create a [`Function`][agentdojo.functions_runtime.Function] instance from.
            The function should have docstrings with a short description of the function and descriptions
            for each argument. [`Depends`][agentdojo.functions_runtime.Depends] arguments should not be
            documented in the docstring.

    Returns:
        A [`Function`][agentdojo.functions_runtime.Function] instance.

    Raises:
        ValueError: If the function has no docstring or short description, or if the arguments docstrings are
            invalid.
    """
    dependencies = _get_dependencies(function)

    if function.__doc__ is None or function.__doc__ == "":
        raise ValueError(f"Function {function.__name__} has no docstring")
    function_docs = parse(function.__doc__)
    if function_docs.short_description is None:
        raise ValueError(f"Function {function.__name__} has no short description")
    description = function_docs.short_description
    if function_docs.long_description is not None:
        description = f"{description}\n{function_docs.long_description}"

    return Function[P, T](
        name=function.__name__,
        description=description.replace("None", "null"),
        parameters=_parse_args(function.__name__, function_docs.params, inspect.signature(function)),
        run=function,
        dependencies=dependencies,
        full_docstring=function.__doc__,
        return_type=get_type_hints(function).get("return"),
    )


BM = TypeVar("BM", bound=BaseModel)


def _parse_args(
    function_name: str,
    params_docs: list[DocstringParam],
    signature: inspect.Signature,
    field_fn: Callable = Field,
    create_model_fn: Callable[..., type[BM]] = create_model,
) -> type[BM]:
    params = {}
    for arg in params_docs:
        arg_name = arg.arg_name
        if arg.description is None:
            raise ValueError(f"Argument {arg_name} has no description")
        arg_type = signature.parameters[arg_name].annotation
        default_value = signature.parameters[arg_name].default
        params[arg_name] = (
            arg_type,
            field_fn(
                description=arg.description,
                default=... if default_value == inspect.Parameter.empty else default_value,
            ),
        )

    return create_model_fn(f"Input schema for `{function_name}`", **params)


class ToolNotFoundError(Exception):
    """Exception raised when the requested tool is not found."""

    ...


class DualGuardMiddleware:
    """Centralized dual-defense middleware for user queries and tool outputs."""

    CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    ATTACK_KEYWORD_PATTERN = re.compile(
        r"(?i)\b("  # case-insensitive blocklist of common injection phrases
        r"ignore(?:\s+all)?\s+previous(?:\s+instructions?)?"
        r"|system\s+prompt"
        r"|developer\s+message"
        r"|jailbreak"
        r"|prompt\s+injection"
        r"|disregard"
        r"|override"
        r"|bypass"
        r"|reveal"
        r"|exfiltrate"
        r")\b"
    )
    VOWEL_PATTERN = re.compile(r"[aeiouAEIOU]")

    def __init__(
        self,
        runtime: FunctionsRuntime | None = None,
        *,
        risk_threshold: float = 0.30,
        session_id: str | None = None,
        enable_inbound_boundary: bool = True,
        enable_outbound_fal: bool = True,
    ) -> None:
        self.runtime = runtime
        self.risk_threshold = risk_threshold
        self.session_id = session_id or f"GUARD_ID_{secrets.token_hex(4).upper()}"
        self.enable_inbound_boundary = enable_inbound_boundary
        self.enable_outbound_fal = enable_outbound_fal
        self.audit_log: list[dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        self.current_query_intents: set[str] = set()
        self.current_trusted_email: str | None = None

    def set_query_context(
        self,
        *,
        trusted_query_intents: set[str] | None = None,
        trusted_email: str | None = None,
    ) -> None:
        self.current_query_intents = trusted_query_intents or set()
        self.current_trusted_email = trusted_email

    def attach_runtime(self, runtime: FunctionsRuntime) -> None:
        """Attach the middleware to a runtime after initialization."""
        self.runtime = runtime

    def wrap_user_query(self, query: str) -> str:
        """Escape and wrap inbound user input in a session-specific XML envelope."""
        if not self.enable_inbound_boundary:
            return query

        escaped_query = html.escape(query, quote=True)
        wrapped_query = (
            f'<dual_guard session="{self.session_id}">'
            f"<user_query>{escaped_query}</user_query>"
            f"</dual_guard>"
        )
        self._record_audit(
            direction="inbound",
            before=query,
            after=wrapped_query,
            function_name=None,
            arguments=None,
            risk_score=None,
        )
        return wrapped_query

    def protect_user_query(self, query: str) -> str:
        """Alias for wrapping user input before it enters the agent loop."""
        return self.wrap_user_query(query)

    def execute(
        self,
        env: TaskEnvironment | None,
        function: str,
        kwargs: Mapping[str, FunctionCallArgTypes],
        raise_on_error: bool = False,
    ) -> tuple[FunctionReturnType, str | None]:
        """Execute a function through the dual guard and return the guarded result."""
        if self.runtime is None:
            raise ValueError("DualGuardMiddleware requires an attached FunctionsRuntime.")

        if not self.enable_outbound_fal:
            return self.runtime._execute_function(env, function, kwargs, raise_on_error=raise_on_error)

        before_text = self._stringify_for_audit(kwargs)
        self._record_audit(
            direction="outbound",
            before=before_text,
            after=None,
            function_name=function,
            arguments=dict(kwargs),
            risk_score=None,
        )

        result, error = self.runtime._execute_function(env, function, kwargs, raise_on_error=raise_on_error)
        if error is not None:
            self._record_audit(
                direction="outbound",
                before=before_text,
                after=self._stringify_for_audit(result),
                function_name=function,
                arguments=dict(kwargs),
                risk_score=None,
                error=error,
            )
            return result, error

        guarded_result, risk_score = self._guard_tool_output(result)
        after_text = self._stringify_for_audit(guarded_result)
        self._record_audit(
            direction="outbound",
            before=self._stringify_for_audit(result),
            after=after_text,
            function_name=function,
            arguments=dict(kwargs),
            risk_score=risk_score,
        )
        return guarded_result, None

    def run_function(
        self,
        env: TaskEnvironment | None,
        function: str,
        kwargs: Mapping[str, FunctionCallArgTypes],
        raise_on_error: bool = False,
    ) -> tuple[FunctionReturnType, str | None]:
        """Compatibility alias that routes function execution through the middleware."""
        return self.execute(env, function, kwargs, raise_on_error=raise_on_error)

    def detect_imperative_tone(self, text: str) -> float:
        """Placeholder semantic detector that returns a risk score between 0 and 1."""
        lowered = text.lower()
        risk_score = 0.0
        if self.ATTACK_KEYWORD_PATTERN.search(lowered):
            risk_score += 0.45
        if re.search(r"\b(?:please\s+)?(?:do|follow|run|send|delete|ignore|override|reveal|explain)\b", lowered):
            risk_score += 0.30
        if re.search(r"\b(?:now|immediately|quickly|urgently|must|required)\b", lowered):
            risk_score += 0.15
        if "!" in text:
            risk_score += 0.05
        if len(text) > 240:
            risk_score += 0.05
        return min(risk_score, 1.0)

    def _guard_tool_output(self, output: FunctionReturnType) -> tuple[FunctionReturnType, float | None]:
        if "forward_content" in self.current_query_intents:
            return output, None

        if isinstance(output, str):
            lexical = self._lexical_sanitize(output)
            risk_score = self.detect_imperative_tone(lexical)
            if risk_score > self.risk_threshold:
                return self._fal_encode(lexical), risk_score
            return lexical, risk_score

        if isinstance(output, BaseModel):
            transformed_model = self._transform_base_model(output)
            return transformed_model, self.detect_imperative_tone(self._stringify_for_audit(transformed_model))

        if isinstance(output, list):
            transformed_list = [self._transform_nested_value(item) for item in output]
            return transformed_list, self.detect_imperative_tone(self._stringify_for_audit(transformed_list))

        if isinstance(output, tuple):
            transformed_tuple = tuple(self._transform_nested_value(item) for item in output)
            return transformed_tuple, self.detect_imperative_tone(self._stringify_for_audit(transformed_tuple))

        if isinstance(output, dict):
            transformed_dict = {key: self._transform_nested_value(value) for key, value in output.items()}
            return transformed_dict, self.detect_imperative_tone(self._stringify_for_audit(transformed_dict))

        output_text = self._lexical_sanitize(str(output))
        risk_score = self.detect_imperative_tone(output_text)
        if risk_score > self.risk_threshold:
            return self._fal_encode(output_text), risk_score
        return output, risk_score

    def _transform_nested_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if "forward_content" in self.current_query_intents:
                return value
            lexical = self._lexical_sanitize(value)
            if self.detect_imperative_tone(lexical) > self.risk_threshold:
                return self._fal_encode(lexical)
            return lexical
        if isinstance(value, list):
            return [self._transform_nested_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._transform_nested_value(item) for item in value)
        if isinstance(value, dict):
            return {key: self._transform_nested_value(item) for key, item in value.items()}
        if isinstance(value, BaseModel):
            return self._transform_base_model(value)
        return value

    def _transform_base_model(self, model: BaseModel) -> BaseModel:
        update_data = {field_name: self._transform_nested_value(value) for field_name, value in model.model_dump().items()}
        return model.model_copy(update=update_data)

    def _lexical_sanitize(self, text: str) -> str:
        without_control_chars = self.CONTROL_CHAR_PATTERN.sub(" ", text)
        without_attack_keywords = self.ATTACK_KEYWORD_PATTERN.sub("[redacted]", without_control_chars)
        return re.sub(r"\s+", " ", without_attack_keywords).strip()

    def _fal_encode(self, text: str) -> str:
        return self.VOWEL_PATTERN.sub("_", text)

    def _stringify_for_audit(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return repr(value)

    def _record_audit(
        self,
        *,
        direction: str,
        before: str,
        after: str | None,
        function_name: str | None,
        arguments: dict[str, Any] | None,
        risk_score: float | None,
        error: str | None = None,
    ) -> None:
        record = {
            "direction": direction,
            "before": before,
            "after": after,
            "function": function_name,
            "arguments": arguments,
            "risk_score": risk_score,
            "error": error,
            "session_id": self.session_id,
        }
        self.audit_log.append(record)
        self.logger.info("DualGuardMiddleware audit record: %s", record)


class FunctionsRuntime:
    """A runtime for functions.

    Args:
        functions: A list of functions that the runtime has access to.
    """

    def __init__(self, functions: Sequence[Function] = [], middleware: DualGuardMiddleware | None = None) -> None:
        self.functions: dict[str, Function] = {f.name: f for f in functions}
        self.middleware = middleware
        if self.middleware is not None:
            self.middleware.attach_runtime(self)

    def __repr__(self) -> str:
        return f"FunctionsRuntime(functions={self.functions})"

    @overload
    def register_function(self, function: Callable[P, T]) -> Callable[P, T]: ...
    @overload
    def register_function(self, function: Function) -> Function: ...

    def register_function(self, function: Callable[P, T] | Function) -> Callable[P, T] | Function:
        """Registers a function with the runtime.

        If `function` is a function, it will be converted to a `Function` instance and registered,
        otherwise it will be registered as is. The conversion is made with [`make_function`][agentdojo.functions_runtime.make_function].

        Args:
            function: The function to register. It can be a function or a `Function` instance. See [`make_function`][agentdojo.functions_runtime.make_function]
                for more information.

        Returns:
            The registered function.
        """
        if isinstance(function, Function):
            self.functions[function.name] = function
            return function
        try:
            f = make_function(function)
        except Exception as e:
            raise Exception(f"Error when registering function {function.__name__} in {function.__module__}.") from e
        self.functions[function.__name__] = f
        return function

    def _execute_nested_calls(
        self, env: TaskEnvironment | None, kwargs: Mapping[str, FunctionCallArgTypes]
    ) -> dict[str, FunctionCallArgTypes]:
        """Executes nested function calls.

        Args:
            env: The environment to extract dependencies from.
            function: The function to run.
            kwargs: The arguments to pass to the function.

        Returns:
            A tuple containing the result of the function and an error message if an error occurred.
        """
        updated_kwargs = {}
        for arg_name, arg_value in kwargs.items():
            if not isinstance(arg_value, FunctionCall):
                updated_kwargs[arg_name] = arg_value
            else:
                # We need to raise the exception to exit early if an error occurs
                # This will be handled by `run_function` depending on the value of `raise_on_error`
                # in the caller. We can safely take the first element of the tuple
                # as the second will always be None (otherwise an exception would be raised)
                updated_kwargs[arg_name], _ = self.run_function(
                    env, arg_value.function, arg_value.args, raise_on_error=True
                )
        return updated_kwargs

    def _execute_function(
        self,
        env: TaskEnvironment | None,
        function: str,
        kwargs: Mapping[str, FunctionCallArgTypes],
        raise_on_error: bool = False,
    ) -> tuple[FunctionReturnType, str | None]:
        """Runs a function with the given arguments.

        If `raise_on_error` is `True`, the function will raise an exception if an error occurs. Otherwise, it will return an error message as the second
        argument. The error message has the format `'ErrorType: ErrorMessage'`.

        Args:
            env: The environment to extract dependencies from. It can be `None` if the function has no dependencies.
            function: The name of the function to run.
            kwargs: The arguments to pass to the function.
            raise_on_error: If `True`, the function will raise an exception if an error occurs. Otherwise, it will return an error message.

        Returns:
            A tuple containing the result of the function and an error message if an error occurred. If no error occurred, the second element will be `None`.

        Raises:
            ValidationError: If the arguments are invalid and do not match the function's input schema.
            ToolNotFoundError: If the requested function is not available.
        """
        if function not in self.functions:
            if raise_on_error:
                raise ToolNotFoundError(f"The requested function `{function}` is not available.")
            return "", f"ToolNotFoundError: The requested function `{function}` is not available."
        f = self.functions[function]

        # Execute nested function calls
        try:
            resolved_kwargs = self._execute_nested_calls(env, kwargs)
        except Exception as e:
            if raise_on_error:
                raise e
            return "", f"{type(e).__name__}: {e}"

        # Validate the arguments
        try:
            args = f.parameters.model_validate(resolved_kwargs).model_dump()
        except ValidationError as e:
            if raise_on_error:
                raise e
            return "", f"ValidationError: {e}"

        # Extract dependencies from the environment
        if env is None and len(f.dependencies) > 0:
            raise ValueError(f"Function {function} has dependencies but no environment was provided.")
        else:
            env_args = {
                arg_name: dependency.extract_dep_from_env(env)  # type: ignore -- If there are no dependencies, this line will not be executed
                for arg_name, dependency in f.dependencies.items()
            }
        kwargs_with_deps = args | env_args

        # Run the function
        try:
            return f(**kwargs_with_deps), None
        except Exception as e:
            if raise_on_error:
                raise e
            return "", f"{type(e).__name__}: {e}"

    def execute(
        self,
        env: TaskEnvironment | None,
        function: str,
        kwargs: Mapping[str, FunctionCallArgTypes],
        raise_on_error: bool = False,
    ) -> tuple[FunctionReturnType, str | None]:
        """Execute a function, routing through middleware when configured."""
        if self.middleware is not None:
            return self.middleware.execute(env, function, kwargs, raise_on_error=raise_on_error)
        return self._execute_function(env, function, kwargs, raise_on_error=raise_on_error)

    def run_function(
        self,
        env: TaskEnvironment | None,
        function: str,
        kwargs: Mapping[str, FunctionCallArgTypes],
        raise_on_error: bool = False,
    ) -> tuple[FunctionReturnType, str | None]:
        """Compatibility wrapper around `execute`."""
        return self.execute(env, function, kwargs, raise_on_error=raise_on_error)

    def protect_user_query(self, query: str) -> str:
        """Wrap inbound user input when dual-guard middleware is configured."""
        if self.middleware is None:
            return query
        return self.middleware.protect_user_query(query)

    def update_functions(self, new_functions: dict[str, Function]) -> None:
        self.functions = new_functions
