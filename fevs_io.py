# fevs_io.py
import pandas as pd

def load_excel(fp: str) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(fp)
    return {name: xls.parse(name) for name in xls.sheet_names}
