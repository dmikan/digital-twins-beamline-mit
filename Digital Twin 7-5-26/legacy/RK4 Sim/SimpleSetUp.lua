simion.workbench_program()

local N_ELECTRODES = 19
current_electrode = 1   -- which electrode is "hot" right now; export_basis sets this

function segment.fast_adjust()
    for i = 1, N_ELECTRODES do
        adj_elect[i] = 0.0
    end
    adj_elect[current_electrode] = 1.0
end
-- >>> END ADDED BLOCK <

function segment.other_actions()
end
