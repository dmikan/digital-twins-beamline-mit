simion.workbench_program()
-- current_electrode = 1   -- which electrode is "hot" right now; export_basis sets this
-- -- basis_export_mode = True   -- flip to false when you want manual fastadj voltages

-- function segment.fast_adjust()
--     -- if basis_export_mode then
--         for i = 1, N_ELECTRODES do
--             adj_elect[i] = 0.0
--         end
--         adj_elect[current_electrode] = 1.0
--     -- end
--     -- when basis_export_mode is false, the segment does nothing,
--     -- leaving whatever adj_elect values were set manually (via fastadj/GUI) untouched

function segment.other_actions()
end
