import pandas as pd
from pathlib import Path


class FieldMap:

    def __init__(self, filename):
        self.filename = Path(filename)
        self.df = self._load_data()

    def _load_data(self):
        """
        Reads the SIMION electric field CSV file.
        """
        df = pd.read_csv(self.filename)

        # Clean column names
        df.columns = df.columns.str.strip().str.lower()

        return df

    def summary(self):
        print("========== FIELD MAP ==========")
        print(f"Number of points : {len(self.df)}")
        print()

        print("Columns:")
        print(self.df.columns.tolist())
        print()

        print("Coordinate ranges:")
        print(f"x : {self.df['x'].min()} -> {self.df['x'].max()}")
        print(f"y : {self.df['y'].min()} -> {self.df['y'].max()}")
        print(f"z : {self.df['z'].min()} -> {self.df['z'].max()}")

        print()

        print("Unique coordinates:")
        print(f"Nx = {self.df['x'].nunique()}")
        print(f"Ny = {self.df['y'].nunique()}")
        print(f"Nz = {self.df['z'].nunique()}")

        print()

        print("Electric field ranges:")
        print(f"Ex : {self.df['ex'].min()} -> {self.df['ex'].max()}")
        print(f"Ey : {self.df['ey'].min()} -> {self.df['ey'].max()}")
        print(f"Ez : {self.df['ez'].min()} -> {self.df['ez'].max()}")

    def head(self, n=10):
        print(self.df.head(n))

field = FieldMap(r"C:\Users\julia\OneDrive\Documents\Hackathon Gemelos Digitales\DraftHackathon\Hackathon_student\Electrode info\RK4 Sim\simion_efield_output.csv")

field.summary()

print(field.df.shape)