"""
dual_grid.py
=============

Las dos piezas del arreglo del punto ciego del cuadrupolo (2026-07-06):

ParedesPA -- indice de paredes desde LA MISMA fuente que el campo.
    Los puntos de metal salen de pa:point() del PA refinado (mascaras en
    basis_quad/), no de los STL con su transformacion manual de residuo
    11-15mm. Duck-type compatible con electrode_geometry.WallIndex (solo
    .distance()). Los electrodos excluidos (fuente 1, tubo 2, detector
    19) se filtran CLASIFICANDO cada punto de mascara por su STL mas
    cercano -- el residuo de 15mm no cambia a que electrodo pertenece un
    voxel (la separacion entre electrodos distintos es mayor), asi que
    la clasificacion es robusta aunque la posicion absoluta no lo sea.

CampoDual -- resolucion doble: bases globales de ~10mm en toda la
    camara + bases finas de 2.5mm en la caja del cuadrupolo (muestreadas
    del MISMO PA de 1mm -- la resolucion ya estaba pagada). field_batch
    rutea cada punto al mapa que corresponde. La caja fina esta paddeada
    25mm mas alla de los rodillos, asi el salto de interpolacion en la
    frontera cae donde el campo ya es suave.

PROMOVIDO A RAIZ el 2026-07-06 tras el mano a mano en dataset identico
(playpen/comparar_porteros.py): empate en la poblacion legada (Spearman
+0.294 vs +0.303; 6/14 vs 5/14 hitters en top-30) y dominio total en la
familia record (388/395 limpias predichas vs 368/384 SIMION -- la fisica
vieja predecia CERO llegada y 85% de choques fantasma para esos mismos
configs). Las bases finas son de 1mm (resolucion nativa del PA), no de
2.5mm como dice arriba -- ver playpen/extract_PA_1mm.lua.

Uso normal (orchestrator y gemelo lo hacen solos):
    from dual_grid import cargar_fisica
    fis = cargar_fisica()   # dual si basis_quad/ esta; si no, la vieja
    fis.bfm, fis.wall, fis.margen, fis.chunk_screening, fis.dual
"""

import collections
import pathlib

import numpy as np
from scipy.spatial import cKDTree

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE

from RK4_sim_basis_batch import BatchBasisFieldMap

EXCLUIR_DEFAULT = (1, 2, 19)   # = orchestrator.WALL_EXCLUDE


class ParedesPA:
    """Indice de distancia-a-metal construido de las mascaras del PA."""

    def __init__(self, puntos_metal, target_spacing=2.0):
        self.puntos = np.asarray(puntos_metal, dtype=float)
        self.target_spacing = target_spacing
        self._tree = cKDTree(self.puntos)

    @classmethod
    def desde_proyecto(cls, root=ROOT, excluir=EXCLUIR_DEFAULT, verbose=True,
                       incluir_corredores=False):
        """incluir_corredores=False por medicion (2026-07-06): densificar
        los nodos de metal en las gargantas SOLO sobre-marca -- con campo
        1mm y margen 0.5, la familia record mide 388/395 limpias SIN las
        mascaras de corredor (SIMION: 368/384) vs 348/375 con ellas, y
        BASE cae de 30 a 15 (real 51)."""
        root = pathlib.Path(root)
        archivos = [root / "basis_quad" / "mask_global_2mm.csv",
                    root / "basis_quad" / "mask_quad_1mm.csv"]
        if incluir_corredores:
            for extra in ("mask_c1_1mm.csv", "mask_c2_1mm.csv"):
                p = root / "basis_quad" / extra
                if p.exists():
                    archivos.append(p)
        pts = np.vstack([np.loadtxt(f, delimiter=",", skiprows=1) for f in archivos])
        if verbose:
            print(f"[ParedesPA] {len(pts)} puntos de metal del PA")

        if excluir:
            # clasificar cada punto por el STL mas cercano; tirar los de
            # los electrodos excluidos
            from electrode_geometry import load_electrode_meshes
            meshes = load_electrode_meshes(root)
            arboles = {}
            for e, mesh in meshes.items():
                muestras, _ = mesh.sample(max(200, int(mesh.area / 25.0)), return_index=True)
                arboles[e] = cKDTree(muestras)
            dist_por_e = np.full((len(meshes), len(pts)), np.inf)
            orden_e = sorted(meshes)
            for j, e in enumerate(orden_e):
                dist_por_e[j], _ = arboles[e].query(pts, workers=-1)
            duenio = np.array(orden_e)[np.argmin(dist_por_e, axis=0)]
            keep = ~np.isin(duenio, list(excluir))
            if verbose:
                for e in excluir:
                    print(f"[ParedesPA]   excluido electrodo {e}: "
                          f"{int((duenio == e).sum())} puntos")
            pts = pts[keep]
        if verbose:
            print(f"[ParedesPA] indice final: {len(pts)} puntos de metal")
        return cls(pts)

    def distance(self, points):
        d, _ = self._tree.query(np.asarray(points, dtype=float), workers=-1)
        return d


class _MapaArrays(BatchBasisFieldMap):
    """BatchBasisFieldMap construido de arrays ya parseados (cache .npz)
    en vez de CSVs -- a 1mm son ~27M de filas y np.loadtxt tardaria
    minutos en cada arranque."""

    def __init__(self, x, y, z, basis_V):
        self.basis_files = []
        self.n_electrodes = basis_V.shape[0]
        self.x, self.y, self.z = x, y, z
        self.Nx, self.Ny, self.Nz = len(x), len(y), len(z)
        self.basis_V = basis_V
        self.dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        self.dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        self.dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
        self.coords = None
        self.voltages = np.zeros(self.n_electrodes)
        self.V = None
        self.E = None
        self._interpolator = None
        self.set_voltages(np.zeros(self.n_electrodes))


def _cargar_caja(root, nombre, archivos):
    """Carga una caja fina con cache .npz (float32; se reconstruye solo
    si algun CSV es mas nuevo que el cache)."""
    cache = root / "basis_quad" / f"cache_{nombre}.npz"
    if cache.exists() and all(cache.stat().st_mtime >= f.stat().st_mtime
                              for f in archivos):
        d = np.load(cache)
        return _MapaArrays(d["x"], d["y"], d["z"], d["basis_V"])

    try:
        import pandas as pd
        leer = lambda f: pd.read_csv(f).to_numpy(dtype=np.float64)
    except ImportError:
        leer = lambda f: np.loadtxt(f, delimiter=",", skiprows=1)

    basis_V = X = Y = Z = None
    for idx, f in enumerate(archivos):
        data = leer(f)
        data = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))]
        x, y, z = (np.unique(data[:, c]) for c in (0, 1, 2))
        if basis_V is None:
            X, Y, Z = x, y, z
            basis_V = np.empty((len(archivos), len(x), len(y), len(z)),
                               dtype=np.float32)
        basis_V[idx] = data[:, 3].reshape(len(x), len(y), len(z)).astype(np.float32)
    np.savez(cache, x=X, y=Y, z=Z, basis_V=basis_V)
    print(f"[CampoDual] cache construido: {cache.name} "
          f"({basis_V.nbytes / 1e6:.0f}MB en float32)")
    return _MapaArrays(X, Y, Z, basis_V)


class CampoDual(BatchBasisFieldMap):
    """Bases globales (heredado) + N cajas de bases finas a lo largo del
    corredor del haz (quad + tubos de los dos tramos). Cada punto se
    rutea a la PRIMERA caja fina que lo contenga; si ninguna, al mapa
    grueso. Diagnostico que motivo los corredores: los contactos
    fantasma del record se concentraban en la garganta del einzel 2
    (x~254-268) -- fuera de la caja del quad, las lentes son el mismo
    filo de navaja."""

    def __init__(self, basis_files, cajas_finas, root=ROOT):
        """cajas_finas: lista de (nombre, [archivos csv por electrodo])."""
        super().__init__(basis_files)
        self.finos = []
        for nombre, archivos in cajas_finas:
            mapa = _cargar_caja(pathlib.Path(root), nombre, archivos)
            lo = np.array([mapa.x.min(), mapa.y.min(), mapa.z.min()])
            hi = np.array([mapa.x.max(), mapa.y.max(), mapa.z.max()])
            self.finos.append((nombre, mapa, lo, hi))
            print(f"[CampoDual] caja '{nombre}': {mapa.Nx}x{mapa.Ny}x{mapa.Nz} "
                  f"(paso {mapa.dx:.1f}mm) x[{lo[0]:g},{hi[0]:g}] "
                  f"y[{lo[1]:g},{hi[1]:g}] z[{lo[2]:g},{hi[2]:g}]")

    @classmethod
    def desde_proyecto(cls, root=ROOT, n_electrodes=19):
        root = pathlib.Path(root)
        gruesos = [root / f"basis_electrode_{i}.csv" for i in range(1, n_electrodes + 1)]
        cajas = []
        for nombre, patron in (("quad", "basis_quad_electrode_{i}.csv"),
                               ("c1", "basis_c1_electrode_{i}.csv"),
                               ("c2", "basis_c2_electrode_{i}.csv")):
            archivos = [root / "basis_quad" / patron.format(i=i)
                        for i in range(1, n_electrodes + 1)]
            if all(f.exists() for f in archivos):
                cajas.append((nombre, archivos))
        return cls(gruesos, cajas)

    # -- voltajes: todos los mapas siempre sincronizados -------------------
    # (finos en float32: a 1mm el E_batch por config son ~26MB en f32 --
    # en f64 duplicaria y el chunk de screening no cabria en RAM)
    def set_voltages_batch(self, voltages_batch, dtype=np.float64):
        out = super().set_voltages_batch(voltages_batch, dtype=dtype)
        for _, mapa, _, _ in self.finos:
            mapa.set_voltages_batch(voltages_batch, dtype=np.float32)
        return out

    # -- consultas: rutear cada punto a la primera caja que lo contenga ----
    def field_batch(self, config_indices, positions):
        E = super().field_batch(config_indices, positions)
        p = np.asarray(positions)
        ci = np.asarray(config_indices)
        pendiente = np.ones(len(p), dtype=bool)
        for _, mapa, lo, hi in self.finos:
            dentro = pendiente & np.all((p >= lo) & (p <= hi), axis=-1)
            if dentro.any():
                E[dentro] = mapa.field_batch(ci[dentro], p[dentro])
                pendiente &= ~dentro
        return E

    def field(self, positions):
        E = super().field(positions)
        p = np.asarray(positions)
        forma = p.shape[:-1]
        plano = p.reshape(-1, 3)
        E = E.reshape(-1, 3)
        pendiente = np.ones(len(plano), dtype=bool)
        for _, mapa, lo, hi in self.finos:
            dentro = pendiente & np.all((plano >= lo) & (plano <= hi), axis=-1)
            if dentro.any():
                E[dentro] = mapa.field(plano[dentro])
                pendiente &= ~dentro
        return E.reshape(*forma, 3)


# ----------------------------------------------------------------------
# EL CARGADOR UNICO -- lo que orchestrator y gemelo llaman
# ----------------------------------------------------------------------
Fisica = collections.namedtuple("Fisica", "bfm wall margen chunk_screening dual")

# margen de contacto calibrado por fisica (barrido 2026-07-06, playpen/
# barrer_margen_pared.py): con paredes del PA el criterio se acerca a
# "entro al metal" (0.5mm); con STL corridos 11-15mm, 1.5mm era el parche.
MARGEN_DUAL = 0.5
MARGEN_STL = 1.5
# los mapas finos de 1mm pesan ~36MB/config en float32: el chunk de
# screening baja de 50 a 8 para no agotar la RAM (mismo trabajo total).
CHUNK_DUAL = 8
CHUNK_STL = 50


def cargar_fisica(root=ROOT, n_electrodes=19, verbose=True):
    """
    Carga la mejor fisica disponible: CampoDual + ParedesPA si los datos
    de basis_quad/ existen (extraidos por playpen/extract_PA_quad.lua +
    extract_PA_1mm.lua), o la fisica clasica (bases 50^3 + paredes STL)
    si no -- asi el codigo corre igual en una maquina que no haya
    extraido las bases finas.
    """
    root = pathlib.Path(root)
    bq = root / "basis_quad"
    tiene_dual = (bq / "mask_global_2mm.csv").exists() and all(
        (bq / f"basis_{caja}_electrode_{i}.csv").exists()
        for caja in ("quad", "c1", "c2") for i in (1, n_electrodes))
    if tiene_dual:
        if verbose:
            print("[fisica] dual 1mm (CampoDual + ParedesPA, margen 0.5)")
        bfm = CampoDual.desde_proyecto(root, n_electrodes=n_electrodes)
        wall = ParedesPA.desde_proyecto(root, verbose=verbose)
        return Fisica(bfm, wall, MARGEN_DUAL, CHUNK_DUAL, True)

    if verbose:
        print("[fisica] clasica (bases 50^3 + STL, margen 1.5) -- "
              "basis_quad/ no encontrado")
    from electrode_geometry import build_wall_index
    bfm = BatchBasisFieldMap.from_directory(root, n_electrodes=n_electrodes)
    wall = build_wall_index(root, exclude=EXCLUIR_DEFAULT, target_spacing=2.0)
    return Fisica(bfm, wall, MARGEN_STL, CHUNK_STL, False)
