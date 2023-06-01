#! /usr/bin/env python

import functools
import itertools
from collections import UserList

from dash import ClientsideFunction, Input, Output, State, dcc
from dash_component_template import ComponentTemplate


class _DepList(UserList):
    # this is used to allow identifying the ensured list
    # so that we know when to automatically wrap the result
    pass


def anysum(a, b):
    """Return the sum of two objects."""
    try:
        return a + b
    except TypeError:
        return itertools.chain(a, b)


def mapreduce(map_func, reduce_func, gen, *args, **kwargs):
    return functools.reduce(reduce_func, map(map_func, gen), *args, **kwargs)


def mapsum(map_func, gen, *args, **kwargs):
    return mapreduce(map_func, anysum, gen, *args, **kwargs)


class SharedDataStore(ComponentTemplate):
    """This class implements a client-side data store that provides data for
    multiple components.

    The `register_callback` method is used to connect the data store
    with inputs and outputs alongside with some callback functions.

    The registered dependencies are collated and actual dash callbacks
    are created internally at the time `setup_layout` is called.
    """

    class Meta:
        component_cls = dcc.Store

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._callbacks = list()

    def register_callback(self, outputs=None, inputs=None, states=None, callback=None):
        """Register a callback."""
        if callback is None:

            def decorator(func):
                return self.register_callback(outputs, inputs, states, callback=func)

            return decorator

        def _ensure_list(lst):
            if lst is None:
                return list()
            if isinstance(lst, (Input, Output, State, str)):
                return _DepList([lst])
            return list(lst)

        self._callbacks.append(
            tuple(map(_ensure_list, (outputs, inputs, states))) + (callback,)
        )

    @staticmethod
    def _make_data_key(output):
        return output.component_id

    @staticmethod
    def _make_unique(items):
        idx = [list() for _ in range(len(items))]
        result = list()
        for i, item in enumerate(items):
            try:
                j = result.index(item)
            except ValueError:
                # not in result yet
                result.append(item)
                idx[i] = len(result) - 1  # the last item
            else:
                # already in result
                idx[i] = j
        return result, idx

    def setup_layout(self, app):
        # this is to hold the key store object that is used to dispatch
        # data in to different output.
        keys = list()
        key_strs = list()
        outputs, idx_o = self._make_unique(mapsum(lambda i: i[0], self._callbacks))
        inputs, idx_i = self._make_unique(mapsum(lambda i: i[1], self._callbacks))
        states, idx_s = self._make_unique(mapsum(lambda i: i[2], self._callbacks))
        # client side
        for output in outputs:
            if isinstance(output, str):
                key_strs.append(output)
                continue
            keys.append(
                self.parent.child(dcc.Store, data=self._make_data_key(output)),
            )
            app.clientside_callback(
                ClientsideFunction(
                    namespace="datastore",
                    function_name="getKey",
                ),
                output,
                [Input(self.id, "data")],
                [State(keys[-1].id, "data")],
            )
        if len(keys) > 0 and len(key_strs) > 0:
            raise ValueError("cannot mix string output with component output")

        # server side
        @app.callback(
            Output(self.id, "data"),
            inputs,
            states + [State(key.id, "data") for key in keys],
        )
        def update(*args):
            # unpack args
            _input_args = args[: len(inputs)]
            _state_args = args[len(inputs) : len(states) - len(keys)]
            if len(keys) > 0:  # component keys
                _key_args = args[len(states) - len(keys) :]
            else:
                _key_args = key_strs

            # de-uniquefy the args
            input_args = [_input_args[i] for i in idx_i]
            state_args = [_state_args[i] for i in idx_s]
            key_args = [_key_args[i] for i in idx_o]

            # dispatch call args
            call_args = list()
            for ii, (o, i, s, c) in enumerate(self._callbacks):
                call_args.append(
                    (
                        key_args[: len(o)],
                        input_args[: len(i)] + state_args[: len(s)],
                        c,
                    )
                )
                key_args = key_args[len(o) :]
                input_args = input_args[len(i) :]
                state_args = state_args[len(s) :]
            # make calls
            result = dict()
            for ii, (k, a, c) in enumerate(call_args):
                r = c(*a)
                if isinstance(self._callbacks[ii][0], _DepList):
                    # the call need to be wrapped as a list
                    r = [r]
                for kk, v in zip(k, r):
                    result[kk] = v
            return result
