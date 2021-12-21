#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 20:54
# @Author  : LiXin
# @File    : semantic_symbol_by_pre.py
# @Describe:
import html
import TangentS.math_tan
from .mathml import MathML
from .math_symbol import MathSymbol
from .exceptions import UnknownTagException

class SemanticSymbolByPre(MathSymbol):
    """
        Symbol in an operator tree
        """
    MaxChildren = 62  # 62
    CommutativePairs = True

    def __init__(self, tag, children=None, parent=None, mathml=None):
        MathSymbol.__init__(self, tag)

        if isinstance(children, list):
            # copy ...
            self.children = list(children)
        else:
            self.children = None

        self.parent = parent
        self.mathml = mathml

    def get_size(self):
        current_size = 1

        if not self.children is None:
            for child in self.children:
                current_size += child.get_size()

        return current_size

    def is_leaf(self):
        return (self.children is None or len(self.children) == 0)

    @staticmethod
    def Copy(other):
        local = SemanticSymbolByPre(other.tag, mathml=other.mathml)

        if other.children is not None:
            local.children = []
            for original_child in other.children:
                copy_child = SemanticSymbolByPre.Copy(original_child)
                copy_child.parent = local
                local.children.append(copy_child)

        return local

    def tree_depth(self):
        if self.children is None or len(self.children) == 0:
            return 1
        else:
            return 1 + max([child.tree_depth() for child in self.children])

    @classmethod
    def parse_from_pre_mathml(cls, elem, parent=None, identified=None):
        retval = None
        if identified is None:
            identified = {}

        short_tag = elem.tag[len(MathML.namespace):]

        # expected MATHML root
        if elem.tag == MathML.math:
            children = list(elem)
            if len(children) == 1:
                retval = cls.parse_from_pre_mathml(children[0], None, identified)
            elif len(children) == 0:
                return None
            else:
                raise Exception('math_tan element with more than 1 child')

        elif elem.tag==MathML.semantics:
            children=list(elem)
            if len(children)>=1:
                return cls.parse_from_pre_mathml(children[0])
            elif len(children)==0:
                return None


        elif elem.tag == MathML.msqrt:
            retval = SemanticSymbolByPre("O!" + short_tag, parent=parent)

        if retval is None:
            raise UnknownTagException(elem.tag)

        if "id" in elem.attrib:
            identified[elem.attrib["id"]] = retval

        if retval.tag[0:2] == "E!":
            # check for common error patterns to simplify tree...

            # contiguous "unknown" csymbol....
            pos = 0
            while pos + 1 < len(retval.children):
                if retval.children[pos].tag[0:2] in ["-!", "T!"] and retval.children[pos + 1].tag[0:2] == "-!":
                    # combine ... change to text ...
                    retval.children[pos].tag = "T!" + retval.children[pos].tag[2:] + retval.children[pos + 1].tag[2:]
                    # remove next ...
                    del retval.children[pos + 1]
                else:
                    pos += 1

            # check ...
            if len(retval.children) > SemanticSymbolByPre.MaxChildren:
                # too many children for a single E! node, split ...
                SemanticSymbolByPre.split_node(retval)

        if (isinstance(retval, SemanticSymbolByPre) and retval.children is not None and
                len(retval.children) > SemanticSymbolByPre.MaxChildren):
            raise Exception("Node exceeds maximum number of childreen allowed (" +
                            str(SemanticSymbolByPre.MaxChildren) + ") - " + str(len(retval.children)))

        return retval

    @staticmethod
    def split_node(node):
        if len(node.children) > SemanticSymbolByPre.MaxChildren:
            # do a binary split
            mid_point = math_tan.ceil(len(node.children) / 2.0)

            # create new parents ...
            left_child = SemanticSymbolByPre(node.tag, children=node.children[:mid_point], parent=node)
            right_child = SemanticSymbolByPre(node.tag, children=node.children[mid_point:], parent=node)

            # link children (now grand-children to their new parents)
            for child in left_child.children:
                child.parent = left_child

            for child in right_child.children:
                child.parent = right_child

            # update node children ...
            node.children = [left_child, right_child]

            # continue splitting recursively ...
            SemanticSymbolByPre.split_node(left_child)
            SemanticSymbolByPre.split_node(right_child)