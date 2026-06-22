"""
Загрузчик матрицы экспрессии генов Xenium
"""
from typing import List, Optional
import h5py
import numpy as np
import scipy.sparse as sp

class XeniumGenePanelLoader:
    def __init__(self, matrix_path: str):
        print(f"🧬 Загрузка матрицы из: {matrix_path}")
        with h5py.File(matrix_path, "r") as f:
            data = f["matrix/data"][:]
            indices = f["matrix/indices"][:]
            indptr = f["matrix/indptr"][:]
            n_features, n_cells = f["matrix/shape"][:]
            
            matrix = sp.csr_matrix(
                (data, indices, indptr), shape=(n_cells, n_features)
            )
            self.matrix = matrix.toarray()
            
            self.gene_names: List[str] = [
                g.decode("utf-8") if isinstance(g, (bytes, bytearray)) else str(g)
                for g in f["matrix/features/name"][:]
            ]
            
            raw_barcodes = f["matrix/barcodes"][:]
            self.cell_ids: List[str] = [
                c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                for c in raw_barcodes
            ]
        
        # Маппинг
        self._id_to_idx = {}
        for i, cid in enumerate(self.cell_ids):
            self._id_to_idx[cid] = i
            if isinstance(cid, str):
                self._id_to_idx[cid.encode('utf-8')] = i
                if cid.endswith('-1'):
                    self._id_to_idx[cid[:-2]] = i
        
        print(f"✅ Матрица {self.matrix.shape} (клетки x гены) загружена. "
              f"Всего клеток: {len(self.cell_ids)}")

    def get_gene_vector(self, cell_id) -> Optional[np.ndarray]:
        if cell_id is None:
            return None
            
        if isinstance(cell_id, (bytes, bytearray)):
            cid_str = cell_id.decode('utf-8')
        else:
            cid_str = str(cell_id).strip()
        
        if cid_str.startswith("b'") and cid_str.endswith("'"):
            cid_str = cid_str[2:-1]
        
        idx = self._id_to_idx.get(cid_str)
        
        if idx is None and cid_str.endswith('-1'):
            idx = self._id_to_idx.get(cid_str[:-2])
        
        if idx is None:
            return None
            
        return self.matrix[idx].astype(np.float32)

    def get_all_gene_names(self) -> List[str]:
        return self.gene_names