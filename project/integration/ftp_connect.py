from ftplib import FTP, error_perm, all_errors
import os
import sys
import subprocess

# ==== CONFIGURATION ====
FTP_HOST = "192.168.7.114"
FTP_USER = "ftpuser"
FTP_PASS = "nandumon"
REMOTE_DIR = "/files"        # Remote FTP folder
LOCAL_DIR = "./downloads"    # Local folder to save files


def download_files():
    try:
        # Connect to FTP server
        ftp = FTP(FTP_HOST, timeout=10)
        ftp.login(FTP_USER, FTP_PASS)
        ftp.set_pasv(True)  # Add this after ftp.login()
        print(f"✅ Connected to {FTP_HOST} as {FTP_USER}")

        try:
            ftp.cwd(REMOTE_DIR)
            print(f"📂 Changed to remote directory: {REMOTE_DIR}")
        except error_perm:
            print(f"❌ Remote directory {REMOTE_DIR} does not exist.")
            ftp.quit()
            return

        # Ensure local folder exists
        os.makedirs(LOCAL_DIR, exist_ok=True)

        try:
            files = ftp.nlst()
            if not files:
                print("⚠️ No files found in remote directory.")
            else:
                print(f"Found {len(files)} files: {files}")
        except all_errors as e:
            print(f"❌ Failed to list files: {e}")
            ftp.quit()
            return

        # Download each file
        for filename in files:
            local_path = os.path.join(LOCAL_DIR, filename)
            try:
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)
                print(f"⬇️  Downloaded: {filename}")
            except all_errors as e:
                print(f"❌ Failed to download {filename}: {e}")

        ftp.quit()
        print("✅ All files processed successfully.")

    except all_errors as e:
        print(f"❌ FTP connection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_files()
    subprocess.run(["python3", "/workspace/integration/detect_tensorrt_async.py"])
