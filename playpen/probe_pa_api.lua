-- probe_pa_api.lua: descubrir resolucion del PA y como detectar metal.
-- Correr desde la raiz del proyecto:
--   simion.exe --nogui lua "playpen/probe_pa_api.lua"

local pa = simion.pas:open('electrode_.PA0')

io.write("=== propiedades del PA ===\n")
for _, k in ipairs({"nx", "ny", "nz", "symmetry", "mirror", "dx_mm", "dy_mm", "dz_mm",
                    "refinable", "filename"}) do
    local ok, v = pcall(function() return pa[k] end)
    io.write(string.format("  pa.%s = %s (ok=%s)\n", k, tostring(v), tostring(ok)))
end

io.write("\n=== metodos de consulta de punto ===\n")
-- punto dentro del rodillo 10 del quad (STL: x[79,110] y[25,78] z[30,62] -> centro ~ (94, 51, 46))
-- y punto en espacio libre (el codo del haz, ~(75, 75, 77))
local candidatos = {
    {"pa:point(94,51,46)",       function() return pa:point(94, 51, 46) end},
    {"pa:solid(94,51,46)",       function() return pa:solid(94, 51, 46) end},
    {"pa:electrode(94,51,46)",   function() return pa:electrode(94, 51, 46) end},
    {"pa:inside(94,51,46)",      function() return pa:inside(94, 51, 46) end},
    {"pa:potential(94,51,46)",   function() return pa:potential(94, 51, 46) end},
    {"pa:potential(75,75,77)",   function() return pa:potential(75, 75, 77) end},
    {"pa:point(75,75,77)",       function() return pa:point(75, 75, 77) end},
}
for _, c in ipairs(candidatos) do
    local nombre, fn = c[1], c[2]
    local ok, a, b, cc = pcall(fn)
    io.write(string.format("  %s -> ok=%s  %s, %s, %s\n", nombre, tostring(ok),
                           tostring(a), tostring(b), tostring(cc)))
end

io.write("\n=== potencial tras fast_adjust unitario del electrodo 10 ===\n")
local adj = {}
for i = 1, 19 do adj[i] = 0.0 end
adj[10] = 1.0
pa:fast_adjust(adj)
io.write(string.format("  dentro rodillo 10 (94,51,46): V=%s\n", tostring(pa:potential(94, 51, 46))))
io.write(string.format("  dentro rodillo 9  (55,48,46): V=%s\n", tostring(pa:potential(55, 48, 46))))
io.write(string.format("  espacio libre     (75,75,77): V=%s\n", tostring(pa:potential(75, 75, 77))))
io.write("listo\n")
