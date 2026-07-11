-- ============================================================
-- extract_PA_corridors.lua
-- ============================================================
-- Extension de extract_PA_quad.lua tras el diagnostico 2026-07-06: los
-- contactos fantasma del config record se concentran en x=254-268 -- la
-- garganta del einzel 2, FUERA de la caja fina del quad. Las lentes son
-- el mismo filo de navaja que el cuadrupolo: campo fuerte en geometria
-- chica que la grilla de 10mm no resuelve, trayectorias infladas que
-- rozan el metal real.
--
-- Solucion: bases finas (2.5mm) en DOS corredores alrededor del eje del
-- haz (el eje va por y=75, z=77 en el tramo -x; x=75, y=75 en el +z):
--   corredor 1 (tramo -x):  x[140,395] y[50,100] z[52,102]
--   corredor 2 (tramo +z):  x[50,100]  y[50,100] z[130,405]
-- Junto con la caja del quad (x[10,140] y[15,110] z[5,130]) cubren el
-- vuelo completo con resolucion fina.
--
-- Correr desde la raiz:  simion.exe --nogui lua "playpen/extract_PA_corridors.lua"
-- ============================================================

local N_ELECTRODES = 19
local pa = simion.pas:open('electrode_.PA0')

local function axis(min_v, max_v, step)
    local pts = {}
    local n = math.floor((max_v - min_v) / step + 1e-9) + 1
    for i = 1, n do pts[i] = min_v + (i - 1) * step end
    return pts
end

local CORREDORES = {
    { nombre = "c1", xs = axis(140.0, 395.0, 2.5), ys = axis(50.0, 100.0, 2.5),
      zs = axis(52.0, 102.0, 2.5) },
    { nombre = "c2", xs = axis(50.0, 100.0, 2.5), ys = axis(50.0, 100.0, 2.5),
      zs = axis(130.0, 405.0, 2.5) },
}

for _, cor in ipairs(CORREDORES) do
    io.write(string.format("corredor %s: %d x %d x %d = %d puntos/electrodo\n",
                           cor.nombre, #cor.xs, #cor.ys, #cor.zs,
                           #cor.xs * #cor.ys * #cor.zs))
end

for e = 1, N_ELECTRODES do
    local adj = {}
    for i = 1, N_ELECTRODES do adj[i] = 0.0 end
    adj[e] = 1.0
    pa:fast_adjust(adj)

    for _, cor in ipairs(CORREDORES) do
        local t0 = os.clock()
        local fname = string.format("basis_quad/basis_%s_electrode_%d.csv", cor.nombre, e)
        local file = io.open(fname, "w")
        if not file then error("no pude abrir " .. fname) end
        file:write("x,y,z,V\n")
        local buf = {}
        for _, x in ipairs(cor.xs) do
            for _, y in ipairs(cor.ys) do
                for _, z in ipairs(cor.zs) do
                    local v = pa:potential(x, y, z)
                    if v == nil then v = 0.0 end
                    buf[#buf + 1] = string.format("%.6f,%.6f,%.6f,%.8f\n", x, y, z, v)
                    if #buf >= 4096 then file:write(table.concat(buf)); buf = {} end
                end
            end
        end
        if #buf > 0 then file:write(table.concat(buf)) end
        file:close()
        io.write(string.format("electrodo %d/%d corredor %s listo (%.1fs)\n",
                               e, N_ELECTRODES, cor.nombre, os.clock() - t0))
    end
end
io.write("CORREDORES COMPLETOS\n")
