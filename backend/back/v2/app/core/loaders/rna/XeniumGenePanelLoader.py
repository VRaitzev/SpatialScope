import numpy as np
import scipy.sparse as sp
import spatialdata as sd
from typing import List, Optional

class XeniumSpatialDataGeneLoader:
    def __init__(self, sdata: sd.SpatialData, table_key: str = "table"):
        """
        Инициализирует загрузчик экспрессии, вытягивая AnnData объект из SpatialData.
        """
        print(f"🧬 Инициализация загрузчика экспрессии из таблицы: '{table_key}'")
        
        if table_key not in sdata.tables:
            raise KeyError(
                f"Таблица '{table_key}' не найдена в контейнере SpatialData. "
                f"Доступные таблицы: {list(sdata.tables.keys())}"
            )
            
        self.table = sdata.tables[table_key]
        
        # 1. Извлекаем матрицу экспрессии генов (.X)
        # В AnnData она часто бывает разреженной (scipy.sparse.csr_matrix).
        # Переводим её в плотный массив (numpy) в памяти для мгновенного доступа O(1).
        if sp.issparse(self.table.X):
            print("⏳ Конвертация разреженной матрицы экспрессии в плотный NumPy-массив...")
            self.matrix = self.table.X.toarray()
        else:
            self.matrix = np.asarray(self.table.X)
            
        # 2. Извлекаем имена генов из индекса столбцов (var_names)
        self.gene_names: List[str] = list(self.table.var_names)
        
        # 3. Извлекаем идентификаторы клеток из индекса строк (obs_names)
        self.cell_ids: List[str] = list(self.table.obs_names)
        
        # 4. Строим хэш-мапу для быстрого поиска строки матрицы по cell_id
        self._id_to_idx = {str(cid): i for i, cid in enumerate(self.cell_ids)}
        
        print(f"✅ Матрица {self.matrix.shape} (клетки x гены) успешно подвязана.")
        print(f"Всего уникальных генов в панели: {len(self.gene_names)}")

    def get_gene_vector(self, cell_id: str) -> Optional[np.ndarray]:
        """
        Возвращает вектор экспрессии генов для конкретной клетки по её cell_id.
        """
        if cell_id is None:
            return None
            
        # Приводим к строке, так как индексы SpatialData стандартизированы как str
        cid_str = str(cell_id).strip()
        
        # Ищем индекс строки в нашей матрице
        idx = self._id_to_idx.get(cid_str)
        
        # В SpatialData суффиксы '-1' обычно уже отрезаны или стандартизированы,
        # но оставим этот фоллбэк на случай расхождения старых версий экспорта
        if idx is None and cid_str.endswith('-1'):
            idx = self._id_to_idx.get(cid_str[:-2])
            
        if idx is None:
            return None
            
        return self.matrix[idx].astype(np.float32)

    def get_all_gene_names(self) -> List[str]:
        """
        Возвращает список всех генов, представленных в панели.
        """
        return self.gene_names