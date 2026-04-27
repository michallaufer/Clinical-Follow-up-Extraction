import runpod
import subprocess
import time
import os

# --- CONFIGURATION ---
# It's safer to set this as an environment variable, but you can paste it here for now
#runpod.api_key = "YOUR_RUNPOD_API_KEY" 
runpod.api_key = os.getenv("RUNPOD_API_KEY")
POD_ID = "ad7d5d831e17" # From your earlier terminal prompt
PORT = "22950"
IP = "94.101.98.218"
SSH_KEY = "~/.ssh/id_ed25519"
LOCAL_DIR = "./Clinical_Follow-up_Extraction/"
REMOTE_DIR = f"root@{IP}:/workspace/"

def run_remote_job():
    # 1. Start the Pod
    print("🚀 Starting Pod... (Billing begins)")
    runpod.start_pod(POD_ID)
    
    # Give the Pod 15 seconds to boot up and initialize SSH
    print("⏳ Waiting for SSH to wake up...")
    time.sleep(15)

    try:
        # 2. Sync Local Code -> Remote Pod
        print("📤 Uploading latest code changes...")
        subprocess.run([
            "scp", "-P", PORT, "-i", SSH_KEY, "-r", 
            LOCAL_DIR, REMOTE_DIR
        ], check=True)

        # 3. Execute the Code
        print("🧠 Executing NLP Extraction on GPU...")
        # Note: we use 'python3' and point to your main script
        subprocess.run([
            "ssh", "-p", PORT, "-i", SSH_KEY, f"root@{IP}", 
            f"cd /workspace/Clinical_Follow-up_Extraction && python3 main.py"
        ], check=True)

        # 4. Download results (e.g., a results.txt or model weights)
        print("📥 Downloading results...")
        subprocess.run([
            "scp", "-P", PORT, "-i", SSH_KEY, 
            f"{REMOTE_DIR}Clinical_Follow-up_Extraction/results.log", "./"
        ], check=False)

    except Exception as e:
        print(f"❌ Error during execution: {e}")
    
    finally:
        # 5. ALWAYS Stop the Pod
        print("🛑 Stopping Pod... (Billing ends)")
        runpod.stop_pod(POD_ID)
        print("✅ Balance saved.")

if __name__ == "__main__":
    run_remote_job()