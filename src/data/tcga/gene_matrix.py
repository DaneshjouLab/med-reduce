"""Gene-level mutation matrix from MAF files.

Builds and manages a gene mutation matrix where:
- Rows are samples (sample_id)
- Columns are genes (Hugo_Symbol)
- Values are 1 (mutated) or 0 (not mutated)

Usage:
    from src.data.tcga import GeneMatrix, GDCClient

    # Build from downloaded MAF files
    client = GDCClient()
    gm = GeneMatrix(client=client)
    gm.build_from_maf_dir(config.maf_dir)
    gm.save(config.tables_dir / "gene_matrix.parquet")

    # Load existing
    gm = GeneMatrix.load(config.tables_dir / "gene_matrix.parquet")

    # Merge with slides
    df = gm.merge(slide_df, genes=["TP53", "KRAS"])
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from src.data.tcga.gdc_client import GDCClient


class GeneMatrix:
    """Manages gene-level mutation matrix from MAF files.

    The matrix is indexed by sample_id (UUID) with genes as columns.
    Values are 1 if the sample has any mutation in that gene, 0 otherwise.

    Note: MAF files contain aliquot UUIDs (Tumor_Sample_UUID), not sample UUIDs.
    This class uses the GDC API to resolve aliquot → sample mapping.
    """

    # Key MAF columns we need
    MAF_ALIQUOT_COL = "Tumor_Sample_UUID"  # This is actually an aliquot UUID
    MAF_GENE_COL = "Hugo_Symbol"

    def __init__(
        self,
        matrix: Optional[pd.DataFrame] = None,
        client: Optional["GDCClient"] = None,
    ):
        """
        Args:
            matrix: Optional pre-built matrix (index=sample_id, columns=genes)
            client: GDCClient for resolving aliquot → sample mapping
        """
        self._matrix = matrix
        self._client = client

    def build_from_maf_dir(self, maf_dir: Path) -> "GeneMatrix":
        """Build gene matrix from all MAF files in directory.

        Expects gdc-client download structure: maf_dir/<uuid>/<filename>.maf.gz

        This method:
        1. Parses MAF files to extract aliquot UUIDs and mutated genes
        2. Queries GDC API to resolve aliquot → sample mapping
        3. Builds matrix indexed by sample_id

        Args:
            maf_dir: Directory containing downloaded MAF files

        Returns:
            self (for chaining)
        """
        maf_dir = Path(maf_dir)

        # Find all MAF files
        maf_files = list(maf_dir.glob("*/*.maf.gz"))
        if not maf_files:
            raise ValueError(f"No MAF files found in {maf_dir}")

        # Step 1: Parse MAF files to get aliquot → genes mapping
        aliquot_genes: Dict[str, set] = {}  # aliquot_id -> set of genes
        for maf_path in maf_files:
            parsed = self._parse_maf_file(maf_path)
            for aliquot_id, genes in parsed.items():
                if aliquot_id not in aliquot_genes:
                    aliquot_genes[aliquot_id] = set()
                aliquot_genes[aliquot_id].update(genes)

        # Step 2: Resolve aliquot → sample mapping via GDC API
        aliquot_to_sample = self._resolve_aliquot_to_sample(list(aliquot_genes.keys()))

        # Step 3: Convert to sample → genes mapping
        sample_genes: Dict[str, set] = {}
        for aliquot_id, genes in aliquot_genes.items():
            sample_id = aliquot_to_sample.get(aliquot_id)
            if sample_id:
                if sample_id not in sample_genes:
                    sample_genes[sample_id] = set()
                sample_genes[sample_id].update(genes)

        if not sample_genes:
            raise ValueError("Could not resolve any aliquot → sample mappings")

        # Step 4: Build matrix
        self._matrix = self._build_matrix(sample_genes)
        return self

    def _parse_maf_file(self, maf_path: Path) -> Dict[str, set]:
        """Parse a single MAF file and extract aliquot-gene pairs.

        Args:
            maf_path: Path to .maf.gz file

        Returns:
            Dict mapping aliquot_id to set of mutated genes
        """
        # Read MAF (tab-separated, may have comment header lines)
        df = pd.read_csv(
            maf_path,
            sep='\t',
            comment='#',
            usecols=[self.MAF_ALIQUOT_COL, self.MAF_GENE_COL],
            dtype=str,
        )

        # Group by aliquot and collect genes
        result: Dict[str, set] = {}
        for aliquot_id, group in df.groupby(self.MAF_ALIQUOT_COL):
            genes = set(group[self.MAF_GENE_COL].dropna().unique())
            result[aliquot_id] = genes

        return result

    def _resolve_aliquot_to_sample(self, aliquot_ids: List[str]) -> Dict[str, str]:
        """Resolve aliquot UUIDs to sample UUIDs via GDC API.

        Args:
            aliquot_ids: List of aliquot UUIDs from MAF files

        Returns:
            Dict mapping aliquot_id to sample_id
        """
        if self._client is None:
            from src.data.tcga.gdc_client import GDCClient
            self._client = GDCClient()

        # Query cases filtered by aliquot_ids, expanding the full path to aliquots
        # Hierarchy: case → sample → portion → analyte → aliquot
        cases = self._client._paginate(
            "cases",
            filters={
                "op": "in",
                "content": {
                    "field": "aliquot_ids",
                    "value": aliquot_ids,
                }
            },
            expand=[
                "samples",
                "samples.portions",
                "samples.portions.analytes",
                "samples.portions.analytes.aliquots",
            ],
        )

        # Build aliquot → sample mapping by walking the hierarchy
        aliquot_to_sample: Dict[str, str] = {}
        for case in cases:
            for sample in case.get("samples", []):
                sample_id = sample.get("sample_id")
                for portion in sample.get("portions", []):
                    for analyte in portion.get("analytes", []):
                        for aliquot in analyte.get("aliquots", []):
                            aliquot_id = aliquot.get("aliquot_id")
                            if aliquot_id and sample_id:
                                aliquot_to_sample[aliquot_id] = sample_id

        return aliquot_to_sample

    def _build_matrix(self, mutations: Dict[str, set]) -> pd.DataFrame:
        """Build sparse-friendly DataFrame from mutations dict.

        Args:
            mutations: Dict mapping sample_id to set of mutated genes

        Returns:
            DataFrame with sample_id as index, genes as columns, 0/1 values
        """
        # Get all unique genes
        all_genes = set()
        for genes in mutations.values():
            all_genes.update(genes)
        all_genes = sorted(all_genes)

        # Build rows
        rows = []
        for sample_id, genes in mutations.items():
            row = {gene: 1 if gene in genes else 0 for gene in all_genes}
            row["sample_id"] = sample_id
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.set_index("sample_id")

        # Convert to sparse dtype for memory efficiency
        for col in df.columns:
            df[col] = pd.arrays.SparseArray(df[col].values, fill_value=0)

        return df

    def save(self, path: Path) -> Path:
        """Save matrix to parquet file.

        Args:
            path: Output path (.parquet)

        Returns:
            Path to saved file
        """
        if self._matrix is None:
            raise ValueError("No matrix to save. Call build_from_maf_dir() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert sparse to dense for parquet (parquet handles compression well)
        dense_df = self._matrix.sparse.to_dense()
        dense_df.to_parquet(path)

        return path

    @classmethod
    def load(cls, path: Path) -> "GeneMatrix":
        """Load matrix from parquet file.

        Args:
            path: Path to .parquet file

        Returns:
            GeneMatrix instance
        """
        path = Path(path)
        df = pd.read_parquet(path)

        # Convert back to sparse
        for col in df.columns:
            df[col] = pd.arrays.SparseArray(df[col].values, fill_value=0)

        return cls(matrix=df)

    def subset(self, genes: List[str]) -> pd.DataFrame:
        """Get matrix for specific genes only.

        Args:
            genes: List of gene names (Hugo symbols)

        Returns:
            DataFrame with only requested genes (missing genes filled with 0)
        """
        if self._matrix is None:
            raise ValueError("No matrix loaded.")

        # Handle genes not in matrix (fill with 0)
        result = pd.DataFrame(index=self._matrix.index)
        for gene in genes:
            if gene in self._matrix.columns:
                result[gene] = self._matrix[gene]
            else:
                result[gene] = 0

        return result

    def merge(
        self,
        slide_df: pd.DataFrame,
        genes: Optional[List[str]] = None,
        how: str = "left",
    ) -> pd.DataFrame:
        """Merge gene mutations into slide table.

        Joins on sample_id. Slides without MAF data get 0 for all genes.

        Args:
            slide_df: DataFrame with sample_id column
            genes: Optional subset of genes to include (None = all genes)
            how: Join type ("left" preserves all slides)

        Returns:
            Slide table with gene columns added
        """
        if self._matrix is None:
            raise ValueError("No matrix loaded.")

        # Get gene data (subset or all)
        if genes is not None:
            gene_df = self.subset(genes)
        else:
            gene_df = self._matrix.sparse.to_dense()

        # Reset index to make sample_id a column for merge
        gene_df = gene_df.reset_index()

        # Merge
        result = slide_df.merge(gene_df, on="sample_id", how=how)

        # Fill NaN with 0 for slides without MAF
        gene_cols = [c for c in gene_df.columns if c != "sample_id"]
        result[gene_cols] = result[gene_cols].fillna(0).astype(int)

        return result

    @property
    def genes(self) -> List[str]:
        """List of all genes in matrix."""
        if self._matrix is None:
            return []
        return list(self._matrix.columns)

    @property
    def samples(self) -> List[str]:
        """List of all samples in matrix."""
        if self._matrix is None:
            return []
        return list(self._matrix.index)

    @property
    def shape(self) -> tuple:
        """Shape of matrix (n_samples, n_genes)."""
        if self._matrix is None:
            return (0, 0)
        return self._matrix.shape

    def to_dataframe(self, dense: bool = True) -> pd.DataFrame:
        """Get the full matrix as a DataFrame.

        Args:
            dense: If True, convert sparse to dense. If False, keep sparse.

        Returns:
            DataFrame with sample_id as index, genes as columns
        """
        if self._matrix is None:
            raise ValueError("No matrix loaded.")
        if dense:
            return self._matrix.sparse.to_dense()
        return self._matrix.copy()

    def __repr__(self) -> str:
        if self._matrix is None:
            return "GeneMatrix(empty)"
        return f"GeneMatrix(samples={len(self.samples)}, genes={len(self.genes)})"
