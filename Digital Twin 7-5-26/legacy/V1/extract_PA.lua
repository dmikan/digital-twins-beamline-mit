
-- This segment runs exactly when the 'fly' command begins
-- function segment.initialize()
    -- Access the first Potential Array instance in the workbench
--     local pa = simion.wb.instances[1].pa
    
--     -- if not pa then
--     --     print("Error: No Potential Array found in workbench instance 1.")
--     --     return
--     -- end
    
--     -- Open a file to dump the grid points (Python will read this)
--     local out_file = io.open("extracted_pa.txt", "w")
--     out_file:write("nx=" .. pa.nx .. " ny=" .. pa.ny .. " nz=" .. pa.nz .. "\n")
--     out_file:write("x,y,z,potential\n")
    
--     -- Loop through the entire 3D grid and extract the solution basis potentials
--     for z = 0, pa.nz - 1 do
--         for y = 0, pa.ny - 1 do
--             for x = 0, pa.nx - 1 do
--                 local volt = pa:potential(x, y, z)
--                 out_file:write(x .. "," .. y .. "," .. z .. "," .. volt .. "\n")
--             end
--         end
--     end
    
--     out_file:close()
--     print("SUCCESS: PA data successfully extracted to extracted_pa.txt")
-- -- end

-- Export basis potential for electrode i
-- run_export.lua
-- Run from command line as:
--   simion.exe --nogui lua run_export.lua

-- ============================================================
-- 1. Use the PA already loaded in the SIMION workbench
-- ============================================================
local N_ELECTRODES = 19

local pa = nil
local pa_source = nil

if simion and simion.wb and simion.wb.instances then
    for i = 1, #simion.wb.instances do
        local inst = simion.wb.instances[i]
        if inst and inst.pa then
            pa = inst.pa
            pa_source = "workbench instance " .. i
            break
        end
    end
end

if not pa then
    error("No potential array was found in the SIMION workbench. Open the PA in the workbench before running this script.")
end

print("Loaded PA from: " .. tostring(pa_source))

-- ============================================================
-- 2. Pull grid parameters out of the PA object
-- ============================================================
nx = pa.nx
ny = pa.ny
nz = pa.nz

-- PA grid spacing in mm. Use defaults if SIMION does not expose them.
dx = tonumber(pa.dx_mm) or 1.0
dy = tonumber(pa.dy_mm) or tonumber(pa.dx_mm) or 1.0
dz = tonumber(pa.dz_mm) or tonumber(pa.dx_mm) or 1.0

n_electrodes = N_ELECTRODES

print(string.format("Grid: %d x %d x %d,  spacing: %.4f x %.4f x %.4f mm",
    nx, ny, nz, dx, dy, dz))
print(string.format("PA methods available: potential_at=%s, potential=%s, field_at=%s, field=%s",
    tostring(type(pa.potential_at) == "function"),
    tostring(type(pa.potential) == "function"),
    tostring(type(pa.field_at) == "function"),
    tostring(type(pa.field) == "function")))

-- ============================================================
-- 3. Wrap SIMION's PA methods as the globals your function expects
-- ============================================================
function potential(x, y, z)
    if not pa then
        error("PA object is nil")
    end
    if x == nil or y == nil or z == nil then
        error("Potential call received nil coordinates")
    end

    if type(pa.potential_at) == "function" then
        return pa:potential_at(x, y, z)
    elseif type(pa.potential) == "function" then
        return pa:potential(x, y, z)
    else
        error("PA object does not expose a usable potential method")
    end
end

function efield(x, y, z)
    if not simion or not simion.wb then
        error("SIMION workbench is not available")
    end
    if x == nil or y == nil or z == nil then
        error("Field call received nil coordinates")
    end

    local ex, ey, ez = simion.wb:efield(x, y, z)
    if ex ~= nil and ey ~= nil and ez ~= nil then
        return ex, ey, ez
    end

    error("Workbench field call returned nil values")
end

-- adj_elect is SIMION's built-in voltage array; it should already exist
-- but initialise it defensively
if not adj_elect then
    adj_elect = {}
    for i = 1, n_electrodes do adj_elect[i] = 0.0 end
end

-- ============================================================
-- 4. Paste your export_basis function here (unchanged)
-- ============================================================
function export_basis(electrode_idx, filename)
    for i = 1, n_electrodes do
        adj_elect[i] = 0.0
    end
    adj_elect[electrode_idx] = 1.0

    local file = io.open(filename, "w")
    if not file then
        error("Could not open output file: " .. filename)
    end

    local header = "x,y,z,ex,ey,ez,v\n"
    file:write(header)
    io.write("=== " .. filename .. " ===\n")

    for xi = 1, nx do
        for yi = 1, ny do
            for zi = 1, nz do
                local x = (xi-1) * dx
                local y = (yi-1) * dy
                local z = (zi-1) * dz
                local ex, ey, ez = efield(x, y, z)
                local v = potential(x, y, z)
                local line = string.format(
                    "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    x, y, z, ex, ey, ez, v
                )
                file:write(line)
            end
        end
    end

    file:close()
    io.write("Saved: " .. filename .. "\n")
end

-- ============================================================
-- 5. Export one file per electrode
-- ============================================================
for i = 1, n_electrodes do
    local fname = string.format("basis_electrode_%d.csv", i)
    io.write(string.format("Exporting electrode %d / %d ...\n", i, n_electrodes))
    export_basis(i, fname)
end

io.write("All basis exports complete.\n")