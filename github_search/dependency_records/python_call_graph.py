from github_search.dependency_records.python_code_analysis import ASTAnalysisResult
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame
import tqdm
from pathlib import Path
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow.parquet as pq


class PythonFilesSchema(pa.DataFrameModel):
    path: str
    content: str
    repo_name: str


class DependencyGraphSchema(pa.DataFrameModel):
    repo_name: str
    target: str
    source: str
    edge_type: str


class GraphExtractor:

    @classmethod
    def extract_repo_dependencies_df(
        cls, files_df: DataFrame[PythonFilesSchema]
    ) -> DataFrame[DependencyGraphSchema]:
        # Generate all edge DataFrames using the helper function
        all_edges = list(cls._generate_edge_dataframes(files_df))

        # Combine all edges
        if all_edges:
            result_df = pd.concat(all_edges, ignore_index=True)
            return result_df
        else:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=["repo_name", "target", "source", "edge_type"])

    @classmethod
    def extract_repo_dependencies_to_parquet(
        cls, files_df: DataFrame[PythonFilesSchema], output_path: Union[str, Path]
    ) -> None:
        """
        Extract repository dependencies and write directly to a parquet file.
        This is more memory efficient than the DataFrame version for large datasets.
        Uses streaming parquet writer to avoid reading/writing entire file repeatedly.
        
        Args:
            files_df: DataFrame containing file paths, content, and repo names
            output_path: Path where the parquet file should be written
        """
        import pyarrow as pa
        
        output_path = Path(output_path)
        
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define schema for the parquet file
        schema = pa.schema([
            ('repo_name', pa.string()),
            ('target', pa.string()),
            ('source', pa.string()),
            ('edge_type', pa.string())
        ])
        
        # Process files in batches and write to parquet using streaming writer
        batch_size = 100  # Process 100 files at a time
        batch_edges = []
        writer = None
        
        try:
            for edge_df in cls._generate_edge_dataframes(files_df):
                batch_edges.append(edge_df)
                
                # Write batch when we reach batch_size
                if len(batch_edges) >= batch_size:
                    writer = cls._write_batch_to_parquet_stream(
                        batch_edges, output_path, schema, writer
                    )
                    batch_edges = []
            
            # Write remaining edges if any
            if batch_edges:
                writer = cls._write_batch_to_parquet_stream(
                    batch_edges, output_path, schema, writer
                )
        finally:
            # Close the writer if it was created
            if writer is not None:
                writer.close()
    
    @classmethod
    def _write_batch_to_parquet_stream(
        cls, batch_edges: list, output_path: Path, schema: "pa.Schema", writer: Optional["pq.ParquetWriter"]
    ) -> "pq.ParquetWriter":
        """
        Write a batch of edge DataFrames to parquet file using streaming writer.
        
        Args:
            batch_edges: List of edge DataFrames to write
            output_path: Path to the parquet file
            schema: PyArrow schema for the parquet file
            writer: Existing ParquetWriter or None for first batch
            
        Returns:
            ParquetWriter instance for subsequent batches
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        if not batch_edges:
            return writer
            
        # Combine batch edges
        batch_df = pd.concat(batch_edges, ignore_index=True)
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(batch_df, schema=schema)
        
        # Create writer on first batch, reuse for subsequent batches
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema)
        
        # Write the batch
        writer.write_table(table)
        
        return writer

    @classmethod
    def _generate_edge_dataframes(cls, files_df: DataFrame[PythonFilesSchema]):
        """
        Generator function that yields edge DataFrames for each file in the input DataFrame.
        
        Args:
            files_df: DataFrame containing file paths, content, and repo names
            
        Yields:
            pd.DataFrame: Edge DataFrames with schema [repo_name, target, source, edge_type]
        """
        from github_search.dependency_records.python_code_analysis import get_ast_analysis_result

        for _, row in tqdm.tqdm(files_df.iterrows(), total=len(files_df)):
            path = row["path"]
            content = row["content"]
            repo_name = row["repo_name"]

            # Extract AST analysis result
            analysis_result = get_ast_analysis_result(
                content, prepend_class_to_method_name=True
            )

            if analysis_result is None:
                # Skip files that couldn't be parsed
                continue

            # Always yield repo-file edge record
            repo_file_edge = pd.DataFrame(
                {
                    "repo_name": [repo_name],
                    "target": [path],
                    "source": [repo_name],
                    "edge_type": ["repo-file"],
                }
            )
            yield repo_file_edge

            # Extract edges for this file
            edges_df = cls.extract_edges_df(path, analysis_result)

            if not edges_df.empty:
                # Add repo column and rename type column to match schema
                edges_df["repo_name"] = repo_name
                edges_df = edges_df.rename(columns={"type": "edge_type"})

                # Reorder columns to match schema
                edges_df = edges_df[["repo_name", "target", "source", "edge_type"]]

                yield edges_df

    @classmethod
    def extract_edges_df(cls, path: str, analysis_result: ASTAnalysisResult):
        """
        Extracts call graph edges for a file with given path:
        - file-class edges (class is defined in this file)
        - file-function edges (function is defined in this file)
        - file-import edges (package is imported in this file)
        - inheritance edges
        - class-method edges (class has this method)
        - function-function (function calls another function)
        - import-import edges (function/class is imported from another file - only uses functions and classes that get called in this file)
        """
        edge_dfs = []

        # File-class edges
        if analysis_result.classes:
            edge_dfs.append(cls._extract_class_edges_df(path, analysis_result.classes))

        # File-function edges
        if analysis_result.functions:
            edge_dfs.append(
                cls._extract_function_edges_df(path, analysis_result.functions)
            )

        # File-import edges
        if analysis_result.imports:
            edge_dfs.append(cls._extract_import_edges_df(path, analysis_result.imports))

        # Inheritance edges
        if analysis_result.inheritance:
            edge_dfs.append(
                cls._extract_inheritance_edges_df(analysis_result.inheritance)
            )

        # Class-method edges
        if analysis_result.methods:
            edge_dfs.append(cls._extract_method_edges_df(analysis_result.methods))

        # Function-function edges
        if analysis_result.function_calls:
            edge_dfs.append(cls._extract_call_edges_df(analysis_result.function_calls))

        # Import-import edges (imported functions/classes that are actually used)
        import_import_edges = cls._extract_import_import_edges_df(analysis_result)
        if not import_import_edges.empty:
            edge_dfs.append(import_import_edges)

        # Combine all edge DataFrames
        if edge_dfs:
            return pd.concat(edge_dfs, ignore_index=True)
        else:
            return pd.DataFrame(columns=["source", "target", "type"])

    @classmethod
    def _extract_class_edges_df(cls, path, classes):
        edges_df = pd.DataFrame({"target": classes})
        edges_df["source"] = path
        edges_df["type"] = "file-class"
        return edges_df

    @classmethod
    def _extract_function_edges_df(cls, path, functions):
        edges_df = pd.DataFrame({"target": functions})
        edges_df["source"] = path
        edges_df["type"] = "file-function"
        return edges_df

    @classmethod
    def _extract_import_edges_df(cls, path, imports):
        from github_search.dependency_records.python_code_analysis import ModuleImport, FromImport

        import_targets = []
        for imp in imports:
            if isinstance(imp, ModuleImport):
                import_targets.append(imp.module_name)
            elif isinstance(imp, FromImport):
                if imp.source_module:
                    import_targets.append(f"{imp.source_module}.{imp.imported_name}")
                else:
                    import_targets.append(imp.imported_name)

        edges_df = pd.DataFrame({"target": import_targets})
        edges_df["source"] = path
        edges_df["type"] = "file-import"
        return edges_df

    @classmethod
    def _extract_inheritance_edges_df(cls, inheritance_edges):
        edges_df = pd.DataFrame(inheritance_edges, columns=["source", "target"])
        edges_df["type"] = "inheritance"
        return edges_df

    @classmethod
    def _extract_method_edges_df(cls, method_edges):
        edges_df = pd.DataFrame(method_edges, columns=["source", "target"])
        edges_df["type"] = "class-method"
        return edges_df

    @classmethod
    def _extract_call_edges_df(cls, call_edges):
        edges_df = pd.DataFrame(call_edges, columns=["source", "target"])
        edges_df["type"] = "function-function"
        return edges_df

    @classmethod
    def _extract_import_import_edges_df(cls, analysis_result: ASTAnalysisResult):
        """
        Extract import-import edges: imported functions/classes that are actually used.
        Creates edges from the imported module/name to the function/class that uses it.
        """
        from github_search.dependency_records.python_code_analysis import ModuleImport, FromImport

        import_import_edges = []

        # Build a set of all called/used names from function calls
        used_names = set()
        for caller, callee in analysis_result.function_calls:
            used_names.add(callee)

        # Check which imports are actually used
        for imp in analysis_result.imports:
            if isinstance(imp, ModuleImport):
                # For "import os", check if "os" or "os.something" is used
                module_name = imp.alias or imp.module_name
                for used_name in used_names:
                    if used_name == module_name or used_name.startswith(
                        f"{module_name}."
                    ):
                        import_import_edges.append((imp.module_name, used_name))

            elif isinstance(imp, FromImport):
                # For "from module import func", check if "func" is used
                imported_name = imp.alias or imp.imported_name
                if imported_name in used_names:
                    if imp.source_module:
                        source = f"{imp.source_module}.{imp.imported_name}"
                    else:
                        source = imp.imported_name
                    import_import_edges.append((source, imported_name))

        if import_import_edges:
            edges_df = pd.DataFrame(import_import_edges, columns=["source", "target"])
            edges_df["type"] = "import-import"
            return edges_df
        else:
            return pd.DataFrame(columns=["source", "target", "type"])
