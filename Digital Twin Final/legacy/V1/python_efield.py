import subprocess
import io
import pandas as pd

def get_simion_dataframe(simion_exe, iob_path, lua_path):
    # Construct the command to run SIMION headlessly (--nogui)
    # It loads the workbench, runs the script, and then exits
    cmd = [
        simion_exe,
        "--nogui",
        iob_path,
        "run",
        lua_path
    ]
    
    # Run the command and capture text output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Separate the background SIMION logs from our raw data lines
    all_lines = result.stdout.splitlines()
    print(all_lines)  # Optional: print all lines for debugging
    data_lines = []
    start_collecting = False
    
    for line in all_lines:
        if "---DATA_START---" in line:
            start_collecting = True
            continue
        if start_collecting and line.strip():
            data_lines.append(line)
            
    # Combine the lines into a single text block
    csv_data = "\n".join(data_lines)
    print(csv_data)  # Optional: print the raw CSV data for debugging
    # Read the text block straight into Pandas without saving a file
    headers = ["x_mm", "y_mm", "z_mm", "ex", "ey", "ez"]
    df = pd.read_csv(io.StringIO(csv_data), names=headers)
    
    return df

# --- Execution Setup ---
# Provide the actual paths to your files
SIMION_PATH = r"C:\Program Files\SIMION-8.1\simion.exe" 
MY_WORKBENCH = r"C:\Users\julia\OneDrive\Documents\Hackathon Gemelos Digitales\DraftHackathon\Hackathon_student\Electrode info\SimpleSetup.iob"
LUA_SCRIPT = r"C:\Users\julia\OneDrive\Documents\Hackathon Gemelos Digitales\DraftHackathon\Hackathon_student\Electrode info\export_efield.lua"

# Fetch the DataFrame directly
df = get_simion_dataframe(SIMION_PATH, MY_WORKBENCH, LUA_SCRIPT)

# Your data is now fully ready in Pandas
print(df.head())
