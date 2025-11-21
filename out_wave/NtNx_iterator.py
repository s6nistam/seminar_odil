import os
import subprocess
import glob
import time
import psutil  # Add this import at the top


def wait_for_children(parent_pid, timeout=30):
    """Wait for all child processes of the given PID to terminate."""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    
    children = parent.children(recursive=True)
    if not children:
        return
    
    # Terminate children gracefully
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    
    # Wait for termination
    gone, alive = psutil.wait_procs(children, timeout=timeout)
    
    # Force kill any remaining processes
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


try:
    os.remove("/home/nico/Desktop/Local/Seminar/seminar_odil/out_wave/errors.txt")
except:
    pass
# Set PYTHONPATH in the environment (used by both scripts)
os.environ["PYTHONPATH"] = "/home/nico/Desktop/Local/Seminar/seminar_odil/src"

clear_command = ["rm", "-f", "u_*.png", "data_*.pickle", "ut_*.png"]

# Define the base command for wave.py (runs in odil/ dir)
# wave_script = "/home/nico/Desktop/Local/Seminar/seminar_odil/examples/wave/wave.py"
# wave_script = "/home/nico/Desktop/Local/Seminar/seminar_odil/examples/wave_fd/wave_fd.py"
wave_script = "/home/nico/Desktop/Local/Seminar/seminar_odil/examples/wave_missing_values/wave.py"
wave_command = [
    "python3", wave_script
]

# Define the GIF script path and command (runs in out_wave/ dir)
# Use 'python3' since it works in the console
gif_script = "/home/nico/Desktop/Local/Seminar/seminar_odil/out_wave/image to gif.py"
gif_command = [
    "python3",  # ✅ This works in console → use it
    gif_script,
    "--output"
]

# List of values (powers of 2 from 4 to 256)
# values = [16, 32, 64, 128, 256]
values = [4, 8, 16, 32, 64, 128, 256]

# Use correct argument names (based on your error, they are --Nt and --Nx)
Nt_arg = "--Nt"
Nx_arg = "--Nx"

# Run the loop
for Nx in values:
    # for Nt in [Nx]:
    for Nt in values:
        if Nt <= Nx:  # Match your pattern: 4,4 8,4 8,8 16,4 16,8 16,16 ...

            # time.sleep(5)
            # CLEAN FILES USING PYTHON'S GLOB (FIXED)
            out_wave_dir = "/home/nico/Desktop/Local/Seminar/seminar_odil/out_wave"
            patterns = ["u_*.png", "data_*.pickle", "ut_*.png"]
            
            print("🧹 Cleaning previous files...")
            files_removed = 0
            for pattern in patterns:
                # Get all matching files in out_wave directory
                files = glob.glob(os.path.join(out_wave_dir, pattern))
                for file_path in files:
                    try:
                        os.remove(file_path)
                        print(f"  Removed: {os.path.basename(file_path)}")
                        files_removed += 1
                    except Exception as e:
                        print(f"  ⚠️ Failed to remove {file_path}: {str(e)}")
            
            if files_removed == 0:
                print("  No files to clean")

            # Build command with correct args
            cmd = wave_command + [
                Nt_arg, str(Nt),
                Nx_arg, str(Nx)
            ]
            

            # Build command with correct args
            cmd = wave_command + [
                Nt_arg, str(Nt),
                Nx_arg, str(Nx)
            ]
            print(f"🚀 Running wave.py: Nt={Nt}, Nx={Nx}")
            proc = subprocess.Popen(
                cmd,
                cwd="/home/nico/Desktop/Local/Seminar/seminar_odil",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            try:
                # Wait for main process to finish
                stdout, stderr = proc.communicate(timeout=None)
                if proc.returncode != 0:
                    print(f"❌ wave.py failed with return code {proc.returncode}")
                    print(f"Stdout: {stdout.decode()}")
                    print(f"Stderr: {stderr.decode()}")
                    # continue
            finally:
                # CRITICAL: Wait for all child processes
                wait_for_children(proc.pid)
                print(f"✅ wave.py and all child processes completed for Nt={Nt}, Nx={Nx}")

            subprocess.run(
                "cut -d ',' -f 9-10 train.csv | tail -1 | awk -F',' '{print \"" + f"Nt{Nt:03d}Nx{Nx:03d} u: " + "\" $1 \" u_fd: \" $2}' >> errors.txt", 
                cwd="/home/nico/Desktop/Local/Seminar/seminar_odil/out_wave",
                check=True,
                shell=True
            )

            # Generate GIF
            output_gif = f"Nt{Nt:03d}Nx{Nx:03d}.gif"
            gif_full_cmd = gif_command + [output_gif]  # No extra quotes!
            
            print(f"🎨 Generating GIF: {' '.join(gif_full_cmd)}")
            try:
                # Run in out_wave/ directory
                subprocess.run(
                    gif_full_cmd,
                    cwd="/home/nico/Desktop/Local/Seminar/seminar_odil/out_wave",
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"❌ GIF generation failed with return code {e.returncode}")
                continue

print("✅ All tasks completed.")