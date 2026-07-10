-- ============================================================
-- extract_PA_1mm.lua
-- ============================================================
-- Opcion 1 de la Task C (2026-07-06): bases a la RESOLUCION NATIVA del
-- PA (1mm) en el corredor completo del haz. El barrido de margen mostro
-- que a 2.5mm la fisica nueva ya ordena vivos/muertos pero BASE queda
-- indistinguible de los muertos y el Spearman archivado no alcanza al
-- viejo (+0.214 vs +0.353) -- el ultimo factor de resolucion pendiente.
--
-- Cajas (cobertura CONTIGUA a lo largo del vuelo):
--   quad: x[28,140]  y[40,100] z[20,130]   (codo + rodillos)
--   c1:   x[140,395] y[50,100] z[52,102]   (tramo -x con las lentes)
--   c2:   x[50,100]  y[50,100] z[130,405]  (tramo +z al detector)
-- SOBREESCRIBE los CSVs de 2.5mm (mismos nombres; regenerables en 2min
-- con extract_PA_quad/corridors.lua si hicieran falta).
-- Tambien: mascaras de metal a 1mm en los corredores (el quad ya tiene).
--
-- Correr desde la raiz:  simion.exe --nogui lua "playpen/extract_PA_1mm.lua"
-- ============================================================

local N_ELECTRODES = 19
local pa = simion.pas:open('electrode_.PA0')

local function axis(min_v, max_v, step)
    local pts = {}
    local n = math.floor((max_v - min_v) / step + 1e-9) + 1
    for i = 1, n do pts[i] = min_v + (i - 1) * step end
    return pts
end

-- v2 (mismo dia): la caja apretada del quad (y[40,100]) fue un error
-- medido -- la familia x0.70 entra ANCHA (sigma 16mm) y sus colas
-- cruzaban a campo grueso a mitad del cuadrupolo (n limpias 250 -> 58).
-- Caja original ancha de vuelta; los corredores c1/c2 ya estan a 1mm y
-- no se regeneran (comentados).
local CAJAS = {
    { nombre = "quad", xs = axis(10.0, 140.0, 1.0), ys = axis(15.0, 110.0, 1.0),
      zs = axis(5.0, 130.0, 1.0) },
    -- { nombre = "c1", xs = axis(140.0, 395.0, 1.0), ys = axis(50.0, 100.0, 1.0),
    --   zs = axis(52.0, 102.0, 1.0) },
    -- { nombre = "c2", xs = axis(50.0, 100.0, 1.0), ys = axis(50.0, 100.0, 1.0),
    --   zs = axis(130.0, 405.0, 1.0) },
}

-- nombres de archivo compatibles con dual_grid.CampoDual.desde_proyecto
local PATRONES = { quad = "basis_quad_electrode_%d.csv",
                   c1 = "basis_c1_electrode_%d.csv",
                   c2 = "basis_c2_electrode_%d.csv" }

for _, caja in ipairs(CAJAS) do
    io.write(string.format("caja %s: %d x %d x %d = %d puntos/electrodo\n",
                           caja.nombre, #caja.xs, #caja.ys, #caja.zs,
                           #caja.xs * #caja.ys * #caja.zs))
end

-- ---- mascaras 1mm de los corredores (el quad ya existe a 1mm) --------
for _, caja in ipairs(CAJAS) do
    if caja.nombre ~= "quad" then
        local t0 = os.clock()
        local fname = string.format("basis_quad/mask_%s_1mm.csv", caja.nombre)
        local file = io.open(fname, "w")
        if not file then error("no pude abrir " .. fname) end
        file:write("x,y,z\n")
        local metal, buf = 0, {}
        for _, x in ipairs(caja.xs) do
            for _, y in ipairs(caja.ys) do
                for _, z in ipairs(caja.zs) do
                    local _, es_metal = pa:point(x, y, z)
                    if es_metal then
                        metal = metal + 1
                        buf[#buf + 1] = string.format("%.0f,%.0f,%.0f\n", x, y, z)
                        if #buf >= 4096 then file:write(table.concat(buf)); buf = {} end
                    end
                end
            end
        end
        if #buf > 0 then file:write(table.concat(buf)) end
        file:close()
        io.write(string.format("%s: %d puntos de metal (%.1fs)\n", fname, metal,
                               os.clock() - t0))
    end
end

-- ---- bases 1mm ------------------------------------------------------
for e = 1, N_ELECTRODES do
    local adj = {}
    for i = 1, N_ELECTRODES do adj[i] = 0.0 end
    adj[e] = 1.0
    pa:fast_adjust(adj)

    for _, caja in ipairs(CAJAS) do
        local t0 = os.clock()
        local fname = string.format("basis_quad/" .. PATRONES[caja.nombre], e)
        local file = io.open(fname, "w")
        if not file then error("no pude abrir " .. fname) end
        file:write("x,y,z,V\n")
        local buf = {}
        for _, x in ipairs(caja.xs) do
            for _, y in ipairs(caja.ys) do
                for _, z in ipairs(caja.zs) do
                    local v = pa:potential(x, y, z)
                    if v == nil then v = 0.0 end
                    buf[#buf + 1] = string.format("%.0f,%.0f,%.0f,%.7g\n", x, y, z, v)
                    if #buf >= 8192 then file:write(table.concat(buf)); buf = {} end
                end
            end
        end
        if #buf > 0 then file:write(table.concat(buf)) end
        file:close()
        io.write(string.format("electrodo %d/%d caja %s listo (%.1fs)\n",
                               e, N_ELECTRODES, caja.nombre, os.clock() - t0))
    end
end
io.write("EXTRACCION 1MM COMPLETA\n")
