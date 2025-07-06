import ast
import json
from typing import List, Tuple, Union, Optional

# You need to install pydantic: pip install pydantic
from pydantic import BaseModel, Field

# --- Pydantic Models for a Structured Result (Updated) ---


class ModuleImport(BaseModel):
    type: str = "module"
    module_name: str
    alias: Optional[str] = None


class FromImport(BaseModel):
    type: str = "from"
    source_module: Optional[str]
    imported_name: str
    alias: Optional[str] = None


class ASTAnalysisResult(BaseModel):
    classes: List[str]
    functions: List[str]
    imports: List[Union[ModuleImport, FromImport]]
    inheritance: List[Tuple[str, str]] = Field(..., alias="inheritance (child, parent)")
    methods: List[Tuple[str, str]] = Field(..., alias="methods (class, method)")
    function_calls: List[Tuple[str, str]] = Field(..., alias="calls (caller, callee)")

    class Config:
        populate_by_name = True


# --- The AST Visitor ---


class ASTExtractor(ast.NodeVisitor):
    def __init__(self, prepend_class_to_method_name: bool = False):
        self.prepend_class_to_method_name = prepend_class_to_method_name
        (
            self.classes,
            self.functions,
            self.raw_imports,
            self.methods,
            self.function_calls,
            self.inheritance_edges,
        ) = ([], [], [], [], [], [])
        self.scope_stack, self.alias_map = [], {}

    def visit_ClassDef(self, node: ast.ClassDef):
        child_class_name = node.name
        self.classes.append(child_class_name)
        for base_node in node.bases:
            parent_class_name = self._get_qualified_name_from_node(base_node)
            if parent_class_name:
                self.inheritance_edges.append((child_class_name, parent_class_name))
        self.scope_stack.append(child_class_name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _get_qualified_name_from_node(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return self.alias_map.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            prefix = self._get_qualified_name_from_node(node.value)
            if prefix == "self":
                return node.attr
            if prefix:
                return f"{prefix}.{node.attr}"
        return None

    def visit_Call(self, node: ast.Call):
        caller_name = self.get_parent_scope()
        if not caller_name:
            self.generic_visit(node)
            return
        callee_name = self._get_qualified_name_from_node(node.func)
        if callee_name:
            self.function_calls.append((caller_name, callee_name))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name_in_code = alias.asname or alias.name
            self.alias_map[name_in_code] = alias.name
            self.raw_imports.append(("import", alias.name, alias.asname))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module
        level = node.level
        # source_module_name = "." * level + (module or "") if level > 0 else module
        for alias in node.names:
            name_in_code = alias.asname or alias.name
            self.alias_map[name_in_code] = alias.name
            self.raw_imports.append(("from", module, alias.name, alias.asname))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        func_name = node.name
        current_scope_name = func_name
        parent_scope = self.get_parent_scope()
        if parent_scope and isinstance(parent_scope, str):
            class_name = parent_scope
            self.methods.append((class_name, func_name))
            if self.prepend_class_to_method_name:
                current_scope_name = f"{class_name}.{func_name}"
        else:
            self.functions.append(func_name)
        self.scope_stack.append(current_scope_name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def get_parent_scope(self):
        return self.scope_stack[-1] if self.scope_stack else None

    def get_results(self) -> ASTAnalysisResult:
        """Processes raw data and returns it as a Pydantic model."""
        parsed_imports = []
        for imp in self.raw_imports:
            if imp[0] == "import":
                parsed_imports.append(ModuleImport(module_name=imp[1], alias=imp[2]))
            elif imp[0] == "from":
                parsed_imports.append(
                    FromImport(source_module=imp[1], imported_name=imp[2], alias=imp[3])
                )

        # --- THE CORRECTED SORTING KEY ---
        def sort_key_for_imports(imp: Union[ModuleImport, FromImport]):
            """
            Creates a homogenous tuple of strings for sorting to prevent TypeErrors.
            The key is (primary_name, secondary_name, alias).
            """
            if isinstance(imp, ModuleImport):
                # For `import os`, key is ('os', '', '')
                return (imp.module_name or "", "", imp.alias or "")
            elif isinstance(imp, FromImport):
                # For `from .utils import helper`, key is ('.utils', 'helper', '')
                return (
                    imp.source_module or "",
                    imp.imported_name or "",
                    imp.alias or "",
                )
            return ("", "", "")  # Should not be reached

        parsed_imports.sort(key=sort_key_for_imports)

        return ASTAnalysisResult(
            classes=sorted(list(set(self.classes))),
            functions=sorted(list(set(self.functions))),
            imports=parsed_imports,
            inheritance=sorted(list(set(self.inheritance_edges))),
            methods=sorted(list(set(self.methods))),
            function_calls=sorted(list(set(self.function_calls))),
        )


# --- Main Analysis Function ---


def get_ast_analysis_result(
    source_code: str, prepend_class_to_method_name: bool = False
) -> Optional[ASTAnalysisResult]:
    try:
        tree = ast.parse(source_code)
        extractor = ASTExtractor(
            prepend_class_to_method_name=prepend_class_to_method_name
        )
        extractor.visit(tree)
        return extractor.get_results()
    except SyntaxError as e:
        print(f"Error parsing the code: {e}")
        return None
