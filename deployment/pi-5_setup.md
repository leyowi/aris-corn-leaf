# ARIS - Raspberry Pi 5 setup (Phase B)

Steps 1 to 8 need only the Pi, your PC and Wi-Fi. Step 9 (speed test) also needs
only the Pi: it uses saved test photos, no camera, GPS or ESP32.

## What you need
- Raspberry Pi 5 (4 GB or 8 GB), official 27 W USB-C power supply
- microSD card, 32 GB or more (A2 class is faster)
- Active Cooler (or a fan case): AI models make the Pi hot; without cooling it slows down
- Your PC on the same Wi-Fi network

## 1. Flash the operating system (on your PC)
1. Install **Raspberry Pi Imager** from raspberrypi.com/software
2. Choose Device: **Raspberry Pi 5**
3. Choose OS: **Raspberry Pi OS Lite (64-bit)** (no desktop, saves memory; you control it from your PC)
   - Must be **64-bit**: the AI libraries do not support 32-bit
4. Choose your microSD card, then **Edit settings** (OS customisation):
   - Hostname: `aris-pi`
   - Username and password: e.g. `aris` / a password you will remember
   - Wi-Fi: your network name and password, country `PH`
   - Time zone: `Asia/Manila`
   - Services tab: **Enable SSH**, use password authentication
5. Write. Takes about 5 minutes.

## 2. First boot and connect from your PC
1. Put the card in the Pi, connect power. Wait about 2 minutes.
2. On your PC (PowerShell or Git Bash):
   ```bash
   ssh aris@aris-pi.local
   ```
   Type `yes` the first time, then your password.
   - If `aris-pi.local` is not found: look up the Pi's IP address in your router's
     device list and use `ssh aris@192.168.x.x` instead.

From now on, commands marked **(Pi)** run in this SSH window.

## 3. Update the system (Pi)
```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```
Wait a minute, then `ssh` in again.

## 4. Install system packages (Pi)
```bash
sudo apt install -y python3-venv python3-pip git libgl1 libglib2.0-0
python3 --version
```

## 5. Get the code (Pi)
The repo is private, so GitHub needs a **personal access token** instead of your password:
1. On GitHub: Settings > Developer settings > Personal access tokens > Fine-grained tokens >
   Generate new token. Repository access: only `aris-corn-leaf`. Permissions: Contents = Read-only.
2. Copy the token (you only see it once).
3. On the Pi:
   ```bash
   cd ~
   git clone https://github.com/leyowi/aris-corn-leaf.git ARIS-CORN
   ```
   Username: your GitHub username. Password: **paste the token**.

Later, to get code updates: `cd ~/ARIS-CORN && git pull`

## 6. Python environment (Pi)
```bash
cd ~/ARIS-CORN
python3 -m venv venv-pi5
source venv-pi5/bin/activate
pip install --upgrade pip
pip install -r requirements-pi5.txt
```
This takes 10 to 20 minutes (PyTorch is large). You will see `(venv-pi5)` at the start
of the prompt. **Activate it every time you log in:** `source ~/ARIS-CORN/venv-pi5/bin/activate`

## 7. Export the models (PC)
On your PC, in the project folder with `venv` active:
```bash
pip install onnxslim ncnn pnnx
python deployment/onnx_export/export_models.py
```
Every line should start with `[ok]`. `[skip] NCNN ...` is acceptable: ONNX still works.
It creates `deployment/bundle/` (about 90 MB): models, test photos, reference answers.

## 8. Copy the bundle to the Pi (PC)
From the project folder on your PC:
```bash
scp -r deployment/bundle aris@aris-pi.local:~/ARIS-CORN/deployment/
```

## 9. Speed test (Pi)
```bash
cd ~/ARIS-CORN
source venv-pi5/bin/activate
python deployment/edge_inference/benchmark_pi.py
```
About 5 to 15 minutes. Check:
- **Same labels as PC** should be close to 38/38. Much lower means an export problem.
- **CPU temp at end** above about 80 C means the Pi is slowing itself down: add cooling.

Bring the report back to your PC (run on the PC):
```bash
scp aris@aris-pi.local:~/ARIS-CORN/results/metrics/pi5_benchmark.md results/metrics/
```

## Shutting down safely (Pi)
```bash
sudo shutdown now
```
Wait for the green light to stop blinking before unplugging. Pulling the power while
running can corrupt the microSD card.

## Common problems
| Message | Fix |
|---|---|
| `externally-managed-environment` | The venv is not active: `source ~/ARIS-CORN/venv-pi5/bin/activate` |
| `libGL.so.1: cannot open shared object` | `sudo apt install -y libgl1` |
| `Killed` during pip install or benchmark | Out of memory: reboot, close other programs, try again |
| `Could not resolve hostname aris-pi.local` | Use the Pi's IP address instead |
| `Permission denied (publickey,password)` | Wrong username or password from step 1 |