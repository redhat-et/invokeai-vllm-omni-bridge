"""Stub out the ``invokeai`` package hierarchy so that node modules can be
imported in the test environment without InvokeAI being installed.

This module is loaded by pytest before any test collection, which means the
fake entries are present in ``sys.modules`` before the first ``import`` of any
node module.
"""

import sys
import types

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stub classes
# ---------------------------------------------------------------------------

class BaseInvocationOutput(BaseModel):
    """Minimal stand-in for InvokeAI's BaseInvocationOutput."""


class BaseInvocation(BaseModel):
    """Minimal stand-in for InvokeAI's BaseInvocation."""


def invocation(type_id: str, **kwargs):
    """Passthrough decorator — just returns the decorated class unchanged."""
    def decorator(cls):
        return cls
    return decorator


def invocation_output(type_id: str):
    """Passthrough decorator — just returns the decorated class unchanged."""
    def decorator(cls):
        return cls
    return decorator


def InputField(**kwargs):
    """Stub for InvokeAI's InputField — delegates to pydantic Field."""
    kwargs.pop("ui_component", None)
    return Field(**kwargs)


def OutputField(**kwargs):
    """Stub for InvokeAI's OutputField — delegates to pydantic Field."""
    return Field(**kwargs)


class UIComponent:
    Textarea = "textarea"


class ImageField(BaseModel):
    """Minimal stand-in for InvokeAI's ImageField (a named image reference)."""

    image_name: str = ""


class InvocationContext:
    """Minimal stand-in for InvokeAI's InvocationContext."""


# ---------------------------------------------------------------------------
# Inject stubs into sys.modules
# ---------------------------------------------------------------------------

def _mod(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


_invokeai = _mod("invokeai")
_app = _mod("invokeai.app")
_inv = _mod("invokeai.app.invocations")
_base = _mod("invokeai.app.invocations.baseinvocation")
_fields = _mod("invokeai.app.invocations.fields")
_svc = _mod("invokeai.app.services")
_svc_shared = _mod("invokeai.app.services.shared")
_ctx = _mod("invokeai.app.services.shared.invocation_context")

_base.BaseInvocation = BaseInvocation
_base.BaseInvocationOutput = BaseInvocationOutput
_base.invocation = invocation
_base.invocation_output = invocation_output

_fields.InputField = InputField
_fields.OutputField = OutputField
_fields.UIComponent = UIComponent
_fields.ImageField = ImageField

_ctx.InvocationContext = InvocationContext
