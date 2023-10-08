"""Utilities for flattening functions' inputs and outputs."""

import itertools
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from typing import Any, NamedTuple

from minjax import utils


class NodeType(NamedTuple):
    name: str
    to_iterable: Callable
    from_iterable: Callable


node_types: dict[type, NodeType] = {
    tuple: NodeType("tuple", lambda xs: (None, xs), lambda _, xs: tuple(xs)),
    list: NodeType("list", lambda xs: (None, xs), lambda _, xs: list(xs)),
    dict: NodeType(
        "dict",
        lambda xs: map(tuple, utils.unzip2(sorted(xs.items()))),
        lambda keys, vals: dict(zip(keys, vals, strict=True)),
    ),
}


class PyTreeDef(NamedTuple):
    node_type: NodeType
    node_metadata: Hashable
    child_treedefs: tuple["PyTreeDef", ...]


class Leaf:
    pass


leaf = Leaf()


def tree_flatten(x: Any) -> tuple[list[Any], PyTreeDef | Leaf]:
    def _tree_flatten(x: Any) -> tuple[Iterable, PyTreeDef | Leaf]:
        node_type = node_types.get(type(x))
        if node_type:
            node_metadata, children = node_type.to_iterable(x)
            children_flat, child_trees = utils.unzip2(
                list(map(_tree_flatten, children))
            )
            flattened = itertools.chain.from_iterable(children_flat)
            return flattened, PyTreeDef(node_type, node_metadata, tuple(child_trees))
        else:
            return [x], leaf

    children_iter, treedef = _tree_flatten(x)
    return list(children_iter), treedef


def tree_unflatten(treedef: PyTreeDef | Leaf, xs: Sequence[Any]) -> Any:
    def _tree_unflatten(treedef: PyTreeDef | Leaf, xs: Iterator) -> Any:
        if isinstance(treedef, Leaf):
            return next(xs)
        else:
            children = (_tree_unflatten(t, xs) for t in treedef.child_treedefs)
            return treedef.node_type.from_iterable(treedef.node_metadata, children)

    return _tree_unflatten(treedef, iter(xs))


class Empty:
    pass


empty = Empty()


class Store:
    val: PyTreeDef | Leaf | Empty = empty

    def set_value(self, val: PyTreeDef | Leaf) -> None:
        if not isinstance(self.val, Empty):
            error_message = "A flatten function cannot be called more than once."
            raise ValueError(error_message)

        self.val = val

    def __call__(self) -> PyTreeDef | Leaf:
        if isinstance(self.val, Empty):
            error_message = "A store must be set with a value before it can be called."
            raise ValueError(error_message)

        return self.val


def flatten_fun(f, in_tree) -> tuple[Callable, Store]:
    store = Store()

    def flat_fun(*args_flat: Any) -> list[Any]:
        pytree_args = tree_unflatten(in_tree, args_flat)
        out = f(*pytree_args)
        out_flat, out_tree = tree_flatten(out)
        store.set_value(out_tree)
        return out_flat

    return flat_fun, store
