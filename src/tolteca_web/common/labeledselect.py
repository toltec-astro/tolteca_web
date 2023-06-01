#! /usr/bin/env python

import dash_bootstrap_components as dbc
from dash_component_template import ComponentTemplate

__all__ = ["LabeledDropdown", "LabeledChecklist"]


class LabeledDropdown(ComponentTemplate):
    """A labeled dropdown widget."""

    class Meta:
        component_cls = dbc.InputGroup

    def __init__(self, label_text, *args, dropdown_props=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_text = label_text
        self.dropdown_props = dropdown_props or dict()

        self.child(dbc.InputGroupText(self.label_text))
        self._dropdown = self.child(dbc.Select, **self.dropdown_props)

    @property
    def dropdown(self):
        """The dropdown component."""
        return self._dropdown


class LabeledChecklist(ComponentTemplate):
    """A labeled checklist widget."""

    class Meta:
        component_cls = dbc.InputGroup

    def __init__(self, label_text, *args, checklist_props=None, multi=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_text = label_text
        self.checklist_props = checklist_props or dict()
        self.multi = multi

        container = self
        label_text = self.label_text
        if not label_text.endswith(":"):
            label_text += ":"
        # container.child(dbc.Label(
        #     label_text, className='mt-2', width='auto',
        #     style={
        #         'color': '#495057',
        #         'font-size': '.875rem',
        #         'padding': '.25rem .5rem',
        #         }
        #     ))
        container.child(
            dbc.InputGroupText(
                label_text, style={"background-color": "#fff", "border": "none"}
            )
        )
        checklist_props = dict(
            labelClassName=(
                "bs4-compat btn btn-sm btn-link" " form-check-label rounded-0"
            ),
            labelCheckedClassName="active btn-outline-primary",
            labelStyle={
                "text-transform": "none",
                "border-style": "none",
            },
            labelCheckedStyle={
                "text-transform": "none",
                "border-style": "none",
            },
            custom=False,
            inputClassName="d-none",
            className="d-flex flex-wrap form-check-compact",
        )
        checklist_props.update(self.checklist_props)
        if self.multi:
            select_cls = dbc.Checklist
        else:
            select_cls = dbc.RadioItems
        # self._checklist = container.child(dbc.Col).child(
        #     select_cls, **checklist_props)
        self._checklist = container.child(select_cls, **checklist_props)

    @property
    def checklist(self):
        """The dropdown component."""
        return self._checklist
