
-- ============================================================
-- 1. Load your potential array file znd load desired SIMION workbench
-- ============================================================

local N_ELECTRODES = 19                 -- <-- update to your actual electrode count


local pa = simion.pas:open('electrode_.PA0')  -- <-- update to your actual PA filename

-- ============================================================
-- 2. Define the evenly spaced sampling grid
-- ============================================================
-- Number of sample points to evaluate potential at along each axis.
-- Change these values to make the export denser or coarser.
local sample_count_x = 50
local sample_count_y = 50
local sample_count_z = 50

-- Bounding box used for the exported potential map.
local x_min, x_max = 0.0, 484.0
local y_min, y_max = 0.0, 153.0
local z_min, z_max = 0.0, 484.0

local function build_axis_points(count, min_value, max_value)
    if count <= 1 then
        return { 0.5 * (min_value + max_value) }
    end

    local points = {}
    local step = (max_value - min_value) / (count - 1)
    for i = 1, count do
        points[i] = min_value + (i - 1) * step
    end
    return points
end

local x_positions = build_axis_points(sample_count_x, x_min, x_max)
local y_positions = build_axis_points(sample_count_y, y_min, y_max)
local z_positions = build_axis_points(sample_count_z, z_min, z_max)

nx = sample_count_x
ny = sample_count_y
nz = sample_count_z

dx = (x_max - x_min) / math.max(1, sample_count_x - 1)
dy = (y_max - y_min) / math.max(1, sample_count_y - 1)
dz = (z_max - z_min) / math.max(1, sample_count_z - 1)

n_electrodes = N_ELECTRODES

-- print(string.format("Sampling grid: %d x %d x %d evenly spaced points", sample_count_x, sample_count_y, sample_count_z))
-- print(string.format("Bounding box: x=%.2f..%.2f, y=%.2f..%.2f, z=%.2f..%.2f", x_min, x_max, y_min, y_max, z_min, z_max))

-- ============================================================
-- 3. Export basis function for a single electrode it takes the electrode index 
--  a pa object and the output filename as arguments 
-- ============================================================
function export_basis(electrode_idx,filename, pa)
    local start_time = os.clock()
    
    current_electrode = electrode_idx
    
    local adj_elect = {}

    for i = 1, N_ELECTRODES do
        adj_elect[i] = 0.0
    end
        adj_elect[current_electrode] = 1.0

    pa:fast_adjust(adj_elect)

    local file = io.open(filename, "w")
    if not file then
        error("Could not open output file: " .. filename)
    end

    local header = "x,y,z,V\n"
    file:write(header)
    -- io.write("=== " .. filename .. " ===\n")

    local xyz_start = os.clock()
    local point_count = 0
    local non_zero_count = 0
    for xi = 1, sample_count_x do
        local x = x_positions[xi]
        for yi = 1, sample_count_y do
            local y = y_positions[yi]
            for zi = 1, sample_count_z do
                local z = z_positions[zi]

                local v = pa:potential(x, y, z)
                if v == nil then v = 0.0 end

                if v ~= 0.0 then
                    non_zero_count = non_zero_count + 1
                end

                local line = string.format(
                    "%.6f,%.6f,%.6f,%.6f\n",
                    x, y, z, v
                )
                file:write(line)
                point_count = point_count + 1
            end
        end
    end
    local xyz_elapsed = os.clock() - xyz_start

    file:close()
    local total_elapsed = os.clock() - start_time
    -- io.write("Saved: " .. filename .. "\n")
    -- io.write(string.format("  XYZ loop time: %.4f seconds (%d points)\n", xyz_elapsed, point_count))
    -- io.write(string.format("  Non-zero potentials found: %d\n", non_zero_count))
    -- io.write(string.format("  Total function time: %.4f seconds\n", total_elapsed))
end

-- ============================================================
-- 5. Export one file per electrode (with timing)
-- ============================================================
local global_start = os.clock()
io.write(string.format("Starting export of %d electrodes...\n", n_electrodes))
io.write(string.format("Grid size: %d x %d x %d = %d total points per electrode\n", nx, ny, nz, nx*ny*nz))
io.write("\n")

for i = 1, n_electrodes do
    local fname = string.format("basis_electrode_%d.csv", i)
    local electrode_start = os.clock()
    io.write(string.format("Exporting electrode %d / %d ...\n", i, n_electrodes))
    export_basis(i, fname, pa)
    local electrode_elapsed = os.clock() - electrode_start
    io.write(string.format("  Electrode %d total time: %.4f seconds\n\n", i, electrode_elapsed))
end

local global_elapsed = os.clock() - global_start
io.write(string.format("All basis exports complete.\n"))
io.write(string.format("TOTAL TIME FOR ALL EXPORTS: %.4f seconds\n", global_elapsed))
io.write(string.format("Average time per electrode: %.4f seconds\n", global_elapsed / n_electrodes))
