-- extract_PA_res25.lua
-- Bases a 2.5mm sobre las MISMAS cajas anchas que las de 1mm, en
-- archivos *25 separados para comparar resolucion sin clobber.
-- simion.exe --nogui lua "playpen/extract_PA_res25.lua"

local N_ELECTRODES = 19
local pa = simion.pas:open('electrode_.PA0')

local function axis(min_v, max_v, step)
    local pts = {}
    local n = math.floor((max_v - min_v) / step + 1e-9) + 1
    for i = 1, n do pts[i] = min_v + (i - 1) * step end
    return pts
end

local S = 2.5
local CAJAS = {
    { nombre = "quad25", xs = axis(10.0, 140.0, S), ys = axis(15.0, 110.0, S), zs = axis(5.0, 130.0, S) },
    { nombre = "c125",   xs = axis(140.0, 395.0, S), ys = axis(50.0, 100.0, S), zs = axis(52.0, 102.0, S) },
    { nombre = "c225",   xs = axis(50.0, 100.0, S), ys = axis(50.0, 100.0, S), zs = axis(130.0, 405.0, S) },
}

for _, c in ipairs(CAJAS) do
    io.write(string.format("caja %s: %d x %d x %d\n", c.nombre, #c.xs, #c.ys, #c.zs))
end

for e = 1, N_ELECTRODES do
    local adj = {}
    for i = 1, N_ELECTRODES do adj[i] = 0.0 end
    adj[e] = 1.0
    pa:fast_adjust(adj)
    for _, caja in ipairs(CAJAS) do
        local fname = string.format("basis_quad/basis_%s_electrode_%d.csv", caja.nombre, e)
        local file = io.open(fname, "w")
        if not file then error("no pude abrir " .. fname) end
        file:write("x,y,z,V\n")
        local buf = {}
        for _, x in ipairs(caja.xs) do
            for _, y in ipairs(caja.ys) do
                for _, z in ipairs(caja.zs) do
                    local v = pa:potential(x, y, z)
                    if v == nil then v = 0.0 end
                    buf[#buf + 1] = string.format("%.4f,%.4f,%.4f,%.8f\n", x, y, z, v)
                    if #buf >= 8192 then file:write(table.concat(buf)); buf = {} end
                end
            end
        end
        if #buf > 0 then file:write(table.concat(buf)) end
        file:close()
    end
    io.write(string.format("electrodo %d/%d listo\n", e, N_ELECTRODES))
end
io.write("2.5mm COMPLETO\n")
