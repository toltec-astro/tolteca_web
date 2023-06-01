#! /usr/bin/env python

import dash_bootstrap_components as dbc
from dash_component_template import ComponentTemplate

__all__ = [
    "LabeledInput",
]


class LabeledInput(ComponentTemplate):
    """A labeled input widget."""

    class Meta:
        component_cls = dbc.InputGroup

    def __init__(self, label_text, *args, input_props=None, suffix_text=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_text = label_text
        input_props = input_props or dict()

        container = self
        container.child(dbc.InputGroupText(self.label_text))
        self._input = container.child(dbc.Input, **input_props)
        if suffix_text is not None:
            container.child(dbc.InputGroupText(suffix_text))
        self._feedback = self.child(dbc.FormFeedback)

    @property
    def input(self):
        """The input component."""
        return self._input

    @property
    def feedback(self):
        """The feedback component."""
        return self._feedback
