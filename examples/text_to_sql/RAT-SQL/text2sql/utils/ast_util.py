#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
# pylint: disable=unused-import
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union
# pylint: enable=unused-import

import asdl
import attr


class ASTWrapperVisitor(asdl.VisitorBase):
    """Used by ASTWrapper to collect information.

    - put constructors in one place.
    - checks that all fields have names.
    - get all optional fields.
    """

    def __init__(self):
        # type: () -> None
        super(ASTWrapperVisitor, self).__init__()
        self.constructors = {}  # type: Dict[str, asdl.Constructor]
        self.sum_types = {}  # type: Dict[str, asdl.Sum]
        self.product_types = {}  # type: Dict[str, asdl.Product]
        self.fieldless_constructors = {}  # type: Dict[str, asdl.Constructor]

    def visitModule(self, mod):
        # type: (asdl.Module) -> None
        for dfn in mod.dfns:
            self.visit(dfn)

    def visitType(self, type_):
        # type: (asdl.Type) -> None
        self.visit(type_.value, str(type_.name))

    def visitSum(self, sum_, name):
        # type: (asdl.Sum, str) -> None
        self.sum_types[name] = sum_
        for t in sum_.types:
            self.visit(t, name)

    def visitConstructor(self, cons, _name):
        # type: (asdl.Constructor, str) -> None
        assert cons.name not in self.constructors
        self.constructors[cons.name] = cons
        if not cons.fields:
            self.fieldless_constructors[cons.name] = cons
        for f in cons.fields:
            self.visit(f, cons.name)

    def visitField(self, field, name):
        # type: (asdl.Field, str) -> None
        # pylint: disable=no-self-use
        if field.name is None:
            raise ValueError(f'Field of type {field.type} in {name} lacks name')

    def visitProduct(self, prod, name):
        # type: (asdl.Product, str) -> None
        self.product_types[name] = prod
        for f in prod.fields:
            self.visit(f, name)


SingularType = Union[asdl.Constructor, asdl.Product]


class ASTWrapper(object):
    """Provides helper methods on the ASDL AST."""

    default_primitive_type_checkers = {
        'identifier': lambda x: isinstance(x, str),
        'int': lambda x: isinstance(x, int),
        'string': lambda x: isinstance(x, str),
        'bytes': lambda x: isinstance(x, bytes),
        'object': lambda x: isinstance(x, object),
        'singleton': lambda x: x is True or x is False or x is None
    }

    # pylint: disable=too-few-public-methods

    def __init__(self, ast_def, custom_primitive_type_checkers={}):
        # type: (asdl.Module, str) -> None
        self.ast_def = ast_def

        visitor = ASTWrapperVisitor()
        visitor.visit(ast_def)

        self.constructors = visitor.constructors
        self.sum_types = visitor.sum_types
        self.product_types = visitor.product_types
        self.seq_fragment_constructors = {}
        self.primitive_type_checkers = {
            **self.default_primitive_type_checkers,
            **custom_primitive_type_checkers
        }
        self.custom_primitive_types = set(custom_primitive_type_checkers.keys())
        self.primitive_types = set(self.primitive_type_checkers.keys())

        # Product types and constructors:
        # no need to decide upon a further type for these.
        self.singular_types = {}  # type: Dict[str, SingularType]
        self.singular_types.update(self.constructors)
        self.singular_types.update(self.product_types)

        # IndexedSets for each sum type
        self.sum_type_vocabs = {
            name: sorted(t.name for t in sum_type.types)
            for name, sum_type in self.sum_types.items()
        }
        self.constructor_to_sum_type = {
            constructor.name: name
            for name, sum_type in self.sum_types.items()
            for constructor in sum_type.types
        }
        self.seq_fragment_constructor_to_sum_type = {
            constructor.name: name
            for name, sum_type in self.sum_types.items()
            for constructor in sum_type.types
        }
        self.fieldless_constructors = sorted(
            visitor.fieldless_constructors.keys())

    @property
    def types(self):
        # type: () -> Dict[str, Union[asdl.Sum, asdl.Product]]
        return self.ast_def.types

    @property
    def root_type(self):
        # type: () -> str
        return self._root_type

    def add_sum_type(self, name, sum_type):
        assert name not in self.sum_types
        self.sum_types[name] = sum_type
        self.types[name] = sum_type

        for type_ in sum_type.types:
            self._add_constructor(name, type_)

    def add_constructors_to_sum_type(self, sum_type_name, constructors):
        for constructor in constructors:
            self._add_constructor(sum_type_name, constructor)
        self.sum_types[sum_type_name].types += constructors

    def remove_product_type(self, product_type_name):
        self.singular_types.pop(product_type_name)
        self.product_types.pop(product_type_name)
        self.types.pop(product_type_name)

    def add_seq_fragment_type(self, sum_type_name, constructors):
        for constructor in constructors:
            # TODO: Record that this constructor is a sequence fragment?
            self._add_constructor(sum_type_name, constructor)

        sum_type = self.sum_types[sum_type_name]
        if not hasattr(sum_type, 'seq_fragment_types'):
            sum_type.seq_fragment_types = []
        sum_type.seq_fragment_types += constructors

    def _add_constructor(self, sum_type_name, constructor):
        assert constructor.name not in self.constructors
        self.constructors[constructor.name] = constructor
        assert constructor.name not in self.singular_types
        self.singular_types[constructor.name] = constructor
        assert constructor.name not in self.constructor_to_sum_type
        self.constructor_to_sum_type[constructor.name] = sum_type_name

        if not constructor.fields:
            self.fieldless_constructors.append(constructor.name)
            self.fieldless_constructors.sort()

    def verify_ast(self, node, expected_type=None, field_path=(), is_seq=False):
        # type: (ASTWrapper, Node, Optional[str], Tuple[str, ...]) -> None
        # pylint: disable=too-many-branches
        """Checks that `node` conforms to the current ASDL."""
        if node is None:
            raise ValueError(f'node is None. path: {field_path}')
        if not isinstance(node, dict):
            raise ValueError(f'node is type {type(node)}. path: {field_path}')

        node_type = node['_type']  # type: str
        if expected_type is not None:
            sum_product = self.types[expected_type]
            if isinstance(sum_product, asdl.Product):
                if node_type != expected_type:
                    raise ValueError(
                        f'Expected type {expected_type}, but instead saw {node_type}. path: {field_path}'
                    )
            elif isinstance(sum_product, asdl.Sum):
                possible_names = [t.name
                                  for t in sum_product.types]  # type: List[str]
                if is_seq:
                    possible_names += [
                        t.name
                        for t in getattr(sum_product, 'seq_fragment_types', [])
                    ]
                if node_type not in possible_names:
                    raise ValueError(
                        f'Expected one of {", ".join(possible_names)}, but instead saw {node_type}. path: {field_path}'
                    )

            else:
                raise ValueError(f'Unexpected type in ASDL: {sum_product}')

        if node_type in self.types:
            # Either a product or a sum type; we want it to be a product type
            sum_product = self.types[node_type]
            if isinstance(sum_product, asdl.Sum):
                raise ValueError(
                    f'sum type {node_type} not allowed as node type. path: {field_path}'
                )
            fields_to_check = sum_product.fields
        elif node_type in self.constructors:
            fields_to_check = self.constructors[node_type].fields
        else:
            raise ValueError(
                f'Unknown node_type {node_type}. path: {field_path}')

        for field in fields_to_check:
            # field.opt:
            # - missing is okay
            # field.seq
            # - missing is okay
            # - otherwise, must be list
            if field.name not in node:
                if field.opt or field.seq:
                    continue
                raise ValueError(
                    f'required field {field.name} is missing. path: {field_path}'
                )

            if field.seq and field.name in node and not isinstance(
                    node[field.name], (list, tuple)):  # noqa: E125
                raise ValueError(
                    f'sequential field {field.name} is not sequence. path: {field_path}'
                )

            # Check that each item in this field has the expected type.
            items = node.get(field.name,
                             ()) if field.seq else (node.get(field.name), )

            # pylint: disable=cell-var-from-loop
            if field.type in self.primitive_type_checkers:
                check = self.primitive_type_checkers[field.type]
            else:
                # pylint: disable=line-too-long
                check = lambda n: self.verify_ast(n,
                                                  field.type,
                                                  field_path + (field.name, ),
                                                  is_seq=field.seq
                                                  )  # noqa: E731,E501

            for item in items:
                assert check(item)
        return True

    def find_all_descendants_of_type(self,
                                     tree,
                                     target_type,
                                     descend_pred=lambda field: True):
        queue = [tree]
        while queue:
            node = queue.pop()
            if not isinstance(node, dict):
                continue
            for field_info in self.singular_types[node['_type']].fields:
                if field_info.opt and field_info.name not in node:
                    continue
                if not descend_pred(field_info):
                    continue

                if field_info.seq:
                    values = node.get(field_info.name, [])
                else:
                    values = [node[field_info.name]]

                if field_info.type == target_type:
                    for value in values:
                        yield value
                else:
                    queue.extend(values)


# Improve this when mypy supports recursive types.
Node = Dict[str, Any]


@attr.s
class HoleValuePlaceholder:
    id = attr.ib()
    is_seq = attr.ib()
    is_opt = attr.ib()
