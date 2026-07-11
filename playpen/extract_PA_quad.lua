-- ============================================================
-- extract_PA_quad.lua
-- ============================================================
-- Resolucion doble para el gemelo RK4 (disenado 2026-07-06):
--
--   1. MASCARA DE METAL desde el propio PA (pa:point -> es_electrodo):
--      la geometria de colision sale de LA MISMA fuente que el campo,
--      eliminando la doble verdad STL-vs-PA (residuos 11-15mm que
--      inventaban 85% de choques fantasma en el cuadrupolo).
--        basis_quad/mask_global_2mm.csv   toda la camara, paso 2mm
--        basis_quad/mask_quad_1mm.csv     caja del quad, paso 1mm
--      (solo se escriben los puntos que SON metal: x,y,z por linea)
--
--   2. BASES FINAS en la caja del cuadrupolo (paso 2.5mm vs ~10mm de
--      las bases globales): el PA subyacente es de 1mm -- la resolucion
--      ya estaba pagada, solo habia que muestrearla donde importa.
--        basis_quad/basis_quad_electrode_N.csv  (x,y,z,V)
--
-- La caja del quad viene de los STL 9-12 (union x[38,110] y[22,81]
-- z[30,102]) + padding que cubre el residuo de alineacion.
--
-- Correr desde la raiz del proyecto (crea basis_quad/ antes):
--   simion.exe --nogui lua "playpen/extract_PA_quad.lua"
-- ============================================================

local N_ELECTRODES = 19
local pa = simion.pas:open('electrode_.PA0')

-- caja de muestreo fino (quad + padding 25mm, recortada a la camara)
local qx0, qx1 = 10.0, 140.0
local qy0, qy1 = 15.0, 110.0
local qz0, qz1 = 5.0, 130.0

-- camara completa (= extract_PA.lua)
local cx0, cx1 = 0.0, 484.0
local cy0, cy1 = 0.0, 153.0
local cz0, cz1 = 0.0, 484.0

local function axis(min_v, max_v, step)
    -- nunca pasarse de max_v: pa:point aborta fuera del PA (y: 0..153)
    local pts = {}
    local n = math.floor((max_v - min_v) / step + 1e-9) + 1
    for i = 1, n do pts[i] = min_v + (i - 1) * step end
    return pts
end

-- ------------------------------------------------------------
-- 1. mascaras de metal (independientes de los voltajes)
-- ------------------------------------------------------------
local function export_mask(filename, xs, ys, zs)
    local t0 = os.clock()
    local file = io.open(filename, "w")
    if not file then error("no pude abrir " .. filename) end
    file:write("x,y,z\n")
    local total, metal = 0, 0
    local buf = {}
    for _, x in ipairs(xs) do
        for _, y in ipairs(ys) do
            for _, z in ipairs(zs) do
                local _, es_metal = pa:point(x, y, z)
                if es_metal then
                    metal = metal + 1
                    buf[#buf + 1] = string.format("%.1f,%.1f,%.1f\n", x, y, z)
                    if #buf >= 4096 then
                        file:write(table.concat(buf)); buf = {}
                    end
                end
                total = total + 1
            end
        end
    end
    if #buf > 0 then file:write(table.concat(buf)) end
    file:close()
    io.write(string.format("%s: %d/%d puntos son metal (%.1fs)\n",
                           filename, metal, total, os.clock() - t0))
end

io.write("=== 1/3: mascara global (2mm) ===\n")
export_mask("basis_quad/mask_global_2mm.csv",
            axis(cx0, cx1, 2.0), axis(cy0, cy1, 2.0), axis(cz0, cz1, 2.0))

io.write("=== 2/3: mascara fina del quad (1mm) ===\n")
export_mask("basis_quad/mask_quad_1mm.csv",
            axis(qx0, qx1, 1.0), axis(qy0, qy1, 1.0), axis(qz0, qz1, 1.0))

-- ------------------------------------------------------------
-- 2. bases finas en la caja del quad (una por electrodo)
-- ------------------------------------------------------------
io.write("=== 3/3: bases finas del quad (2.5mm) ===\n")
local fx = axis(qx0, qx1, 2.5)
local fy = axis(qy0, qy1, 2.5)
local fz = axis(qz0, qz1, 2.5)
io.write(string.format("grilla fina: %d x %d x %d = %d puntos/electrodo\n",
                       #fx, #fy, #fz, #fx * #fy * #fz))

for e = 1, N_ELECTRODES do
    local t0 = os.clock()
    local adj = {}
    for i = 1, N_ELECTRODES do adj[i] = 0.0 end
    adj[e] = 1.0
    pa:fast_adjust(adj)

    local fname = string.format("basis_quad/basis_quad_electrode_%d.csv", e)
    local file = io.open(fname, "w")
    if not file then error("no pude abrir " .. fname) end
    file:write("x,y,z,V\n")
    local buf = {}
    for _, x in ipairs(fx) do
        for _, y in ipairs(fy) do
            for _, z in ipairs(fz) do
                local v = pa:potential(x, y, z)
                if v == nil then v = 0.0 end
                buf[#buf + 1] = string.format("%.6f,%.6f,%.6f,%.8f\n", x, y, z, v)
                if #buf >= 4096 then file:write(table.concat(buf)); buf = {} end
            end
        end
    end
    if #buf > 0 then file:write(table.concat(buf)) end
    file:close()
    io.write(string.format("electrodo %d/%d listo (%.1fs)\n", e, N_ELECTRODES, os.clock() - t0))
end
io.write("EXTRACCION COMPLETA\n")
