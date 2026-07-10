-- extract_wall_masks.lua (2026-07-06)
-- Mascaras de PARED pre-excluidas, con identidad de electrodo del PROPIO
-- PA (no STL): pone los electrodos a CONSERVAR en 1.0 y los excluidos
-- (1 fuente, 2 tubo, 19 detector) en 0.0; un voxel es pared sii
-- is_electrode AND potential>0.5. Elimina la ultima dependencia del STL
-- desalineado (~25mm en el quad, medido).
-- simion.exe --nogui lua "playpen/extract_wall_masks.lua"

local N = 19
local pa = simion.pas:open('electrode_.PA0')
local EXCLUIR = { [1] = true, [2] = true, [19] = true }

local adj = {}
for i = 1, N do adj[i] = EXCLUIR[i] and 0.0 or 1.0 end
pa:fast_adjust(adj)

local function axis(min_v, max_v, step)
    local pts = {}
    local n = math.floor((max_v - min_v) / step + 1e-9) + 1
    for i = 1, n do pts[i] = min_v + (i - 1) * step end
    return pts
end

local function export(fname, xs, ys, zs)
    local file = io.open(fname, "w")
    if not file then error("no pude abrir " .. fname) end
    file:write("x,y,z\n")
    local pared, total, buf = 0, 0, {}
    for _, x in ipairs(xs) do
        for _, y in ipairs(ys) do
            for _, z in ipairs(zs) do
                local v, is_e = pa:point(x, y, z)
                if is_e and v ~= nil and v > 0.5 then
                    pared = pared + 1
                    buf[#buf + 1] = string.format("%.1f,%.1f,%.1f\n", x, y, z)
                    if #buf >= 4096 then file:write(table.concat(buf)); buf = {} end
                end
                total = total + 1
            end
        end
    end
    if #buf > 0 then file:write(table.concat(buf)) end
    file:close()
    io.write(string.format("%s: %d puntos de pared / %d\n", fname, pared, total))
end

io.write("global 2mm...\n")
export("basis_quad/mask_walls_global_2mm.csv", axis(0, 484, 2.0), axis(0, 152, 2.0), axis(0, 484, 2.0))
io.write("quad 1mm...\n")
export("basis_quad/mask_walls_quad_1mm.csv", axis(10, 140, 1.0), axis(15, 110, 1.0), axis(5, 130, 1.0))
io.write("MASCARAS DE PARED COMPLETAS\n")
