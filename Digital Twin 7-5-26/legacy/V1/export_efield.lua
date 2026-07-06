-- Define the bounding box and step size (in mm or grid units)
local x_min, x_max, x_step = 0, 484,  5   -- min mm, max mm, step mm
local y_min, y_max, y_step = 0, 153,  5   -- min mm, max mm, step mm
local z_min, z_max, z_step = 0, 484,  5    -- Keep min/max same for 2D plane

-- Open the output CSV file
local file = assert(io.open("simion_efield_output.csv", "w"))

-- Write the CSV header
file:write("x,y,z,ex,ey,ez\n")

print("Exporting E-field data... please wait.")

-- Loop through the 3D space
for x = x_min, x_max, x_step do
   for y = y_min, y_max, y_step do
      for z = z_min, z_max, z_step do
         
         -- Native SIMION call to get E-field components
         local ex, ey, ez = simion.wb:efield(x, y, z)
         
         -- Handle cases where points are outside the volume (returns nil)
         if ex then
            file:write(string.format("%f,%f,%f,%f,%f,%f\n", x, y, z, ex, ey, ez))
         else
            file:write(string.format("%f,%f,%f,0,0,0\n", x, y, z))
         end
         
      end
   end
end

file:close()
print("Export complete! Saved to simion_efield_output.csv")
