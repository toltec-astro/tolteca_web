import dash_bootstrap_components as dbc
from dash_component_template import ComponentTemplate

__all__ = ["LabeledDropdown", "LabeledChecklist"]


_input_group_text_props = {
    "className": "align-items-baseline",
}


class LabeledDropdown(ComponentTemplate):
    """A labeled dropdown widget."""

    class Meta:  # noqa: D106
        component_cls = dbc.InputGroup

    def __init__(
        self,
        label_text,
        label_first=True,
        *args,
        dropdown_props=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_text = label_text
        self.dropdown_props = dropdown_props or {}
        text_props = _input_group_text_props
        if label_first:
            self.child(dbc.InputGroupText(self.label_text, **text_props))
            self._dropdown = self.child(dbc.Select, **self.dropdown_props)
        else:
            self._dropdown = self.child(dbc.Select, **self.dropdown_props)
            self.child(dbc.InputGroupText(self.label_text, **text_props))
        self._feedback = self.child(dbc.FormFeedback)

    @property
    def dropdown(self):
        """The dropdown component."""
        return self._dropdown

    @property
    def feedback(self):
        """The feedback component."""
        return self._feedback


class LabeledChecklist(ComponentTemplate):
    """A labeled checklist widget."""

    class Meta:  # noqa: D106
        component_cls = dbc.InputGroup

    def __init__(self, label_text, *args, checklist_props=None, multi=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_text = label_text
        self.checklist_props = checklist_props or {}
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
        text_props = _input_group_text_props | {
            "style": {
                "background-color": "#fff",
                "border": "none",
            },
        }
        container.child(
            dbc.InputGroupText(
                label_text,
                **text_props,
            ),
        )
        checklist_props = {
            "labelClassName": (
                "bs4-compat btn btn-sm btn-link form-check-label rounded-0",
            ),
            "labelCheckedClassName": "active btn-outline-primary",
            "labelStyle": {
                "text-transform": "none",
                "border-style": "none",
            },
            "labelCheckedStyle": {
                "text-transform": "none",
                "border-style": "none",
            },
            "custom": False,
            "inputClassName": "d-none",
            "className": "d-flex flex-wrap form-check-compact",
        }
        checklist_props.update(self.checklist_props)
        select_cls = dbc.Checklist if self.multi else dbc.RadioItems
        # self._checklist = container.child(dbc.Col).child(
        #     select_cls, **checklist_props)
        self._checklist = container.child(select_cls, **checklist_props)
        self._feedback = self.child(dbc.FormFeedback)

    @property
    def checklist(self):
        """The dropdown component."""
        return self._checklist

    @property
    def feedback(self):
        """The feedback component."""
        return self._feedback
