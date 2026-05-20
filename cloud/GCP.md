# GCP L4 Instance Setup for H-LeWM

Setup guide for the **H-LeWM** project (Hierarchical Planning with End-to-End JEPA World Models, CS231N). This document walks through provisioning a private GCP Deep Learning VM with an NVIDIA L4 GPU, locking it down from the public internet, providing safe outbound access via Cloud NAT, connecting via IAP-tunneled SSH, and verifying Docker + GPU access for containerized training.

---

## Table of Contents

1. [Create the VM via the GCP Marketplace](#1-create-the-vm-via-the-gcp-marketplace)
2. [Lock Down the VM (Remove External IP)](#2-lock-down-the-vm-remove-external-ip)
3. [Configure Outbound Access via Cloud NAT](#3-configure-outbound-access-via-cloud-nat)
4. [SSH into the Private VM via IAP](#4-ssh-into-the-private-vm-via-iap)
5. [Verify NAT and Internet Connectivity](#5-verify-nat-and-internet-connectivity)
6. [SSH from Local PC](#6-ssh-from-local-pc)
7. [Install Docker](#7-install-docker)
8. [Verify GPU Access Inside Docker](#8-verify-gpu-access-inside-docker)

---

## 1. Create the VM via the GCP Marketplace

The Deep Learning VM image comes preconfigured with CUDA, cuDNN, PyTorch, and the NVIDIA Container Toolkit, which saves a significant amount of setup time.

1. Open the [Google Cloud Console Marketplace](https://console.cloud.google.com/marketplace).
2. Search for **"Deep Learning VM"** and select the official image published by Google.
3. Click **Launch**.
4. Configure the instance:
   - **Framework**: PyTorch (latest with CUDA, e.g., `PyTorch 2.x`).
   - **Machine Type & GPU**: NVIDIA **L4**.
   - **Networking**: Under the External IP dropdown, select **None** so the VM is private from day one.
5. Click **Deploy**.

---

## 2. Lock Down the VM (Remove External IP)

If the VM was created with an external IP, remove it now. A private VM is invisible to the public internet but cannot reach external services like W&B until Cloud NAT is configured (next step).

1. Open the **GCP Console**.
2. In the top search bar, type `VM instances` and open the page.
3. Click the **name** of the VM to open its details page.
4. Click **Edit** at the top.
5. Scroll to **Network interfaces**.
6. Open the dropdown under **External IPv4 address** and change it from `Ephemeral` to `None`.
7. Scroll to the bottom and click **Save**.

The VM is now fully private. It cannot be reached from the internet, and at this stage it cannot reach out either — that is fixed next.

---

## 3. Configure Outbound Access via Cloud NAT

Cloud NAT provides a one-way tunnel for the VM to reach the internet (for W&B logging, `pip install`, dataset downloads, etc.) without exposing it to inbound traffic.

1. In the GCP search bar, type `Cloud NAT` and open it.
2. Click **Create NAT gateway**.
3. Fill out the form:
   - **Gateway name**: `wandb-nat-gateway`
   - **Network**: `default`
   - **Region**: the same region as the VM (e.g., `us-central1`)
   - **Cloud Router**: open the dropdown, choose **Create new router**, name it `nat-router`, and click **Create**.
4. Leave all other settings at their defaults.
5. Click **Create**.

The VM can now talk to W&B, PyPI, GitHub, and other external services while remaining invisible to inbound traffic.

---

## 4. SSH into the Private VM via IAP

Standard SSH will fail since there is no public IP. Google's **Identity-Aware Proxy (IAP)** provides a secure tunnel.

From the GCP console:

1. Open **VM instances**.
2. In the VM's row, click the **SSH** button on the right.
3. Google opens a browser-based terminal tunneled through IAP.

A local-machine SSH workflow is covered in [Section 6](#6-ssh-from-local-pc).

---

## 5. Verify NAT and Internet Connectivity

Run these checks **inside the VM** to confirm everything is wired up.

### 5.1 Confirm Internet Reachability

```bash
curl -I https://google.com
```

- **Success**: response begins with `HTTP/2 200`.
- **Failure**: command hangs or shows "Could not resolve host" — Cloud NAT or VPC routing is misconfigured.

### 5.2 Confirm Traffic Is Masked by the NAT IP

```bash
curl ifconfig.me
```

This prints the public IP the internet sees. To confirm it belongs to Cloud NAT:

1. Go to **Network Services > Cloud NAT** in the console.
2. Click on `wandb-nat-gateway`.
3. Compare the IP from the terminal against the IPs listed under **Cloud NAT IP addresses**. They should match.

### 5.3 Confirm Reachability to Weights & Biases

```bash
curl -I https://wandb.ai
```

An `HTTP/2 200` (or `HTTP/1.1 200 OK`) means the environment is ready for training and metric logging.

---

## 6. SSH from Local PC

For day-to-day work, SSH directly from a local terminal using the `gcloud` CLI with IAP tunneling.

### 6.1 Install and Authenticate the gcloud CLI

1. Install from the [Google Cloud CLI documentation](https://cloud.google.com/sdk/docs/install).
2. Authenticate:

   ```bash
   gcloud auth login
   ```

3. Set the active project (replace `your-project-id`):

   ```bash
   gcloud config set project your-project-id
   ```

### 6.2 Create the IAP Firewall Rule (One-Time Setup)

Allow Google's IAP service to reach port 22. The source range `35.235.240.0/20` is a fixed internal Google IP range used exclusively for IAP tunnels — it does not expose the VM to the public internet.

```bash
gcloud compute firewall-rules create allow-ssh-ingress-from-iap \
    --direction=INGRESS \
    --action=allow \
    --rules=tcp:22 \
    --source-ranges=35.235.240.0/20
```

### 6.3 Connect via IAP from Local Terminal

```bash
gcloud compute ssh ubuntu@VM_NAME --zone=VM_ZONE --tunnel-through-iap
```

Replace:
- `VM_NAME` — name of the instance.
- `VM_ZONE` — VM's zone (e.g., `us-central1-a`).

`ubuntu` is the default user on GCP Deep Learning VMs. Without specifying it, `gcloud` uses your local PC or Google account username, which may not exist on the VM.

### 6.4 Add Aliases to Local ~/.bashrc

Add these to your **local** `~/.bashrc` to avoid typing the full `gcloud` command every time:

```bash
# GCP helpers — set these to your actual VM name and zone
export GCP_VM=<your-vm-name>
export GCP_ZONE=<your-zone>   # e.g. us-central1-a
export GCP_USER=ubuntu        # default user on GCP Deep Learning VMs

alias gcp-ssh='gcloud compute ssh $GCP_USER@$GCP_VM --zone=$GCP_ZONE --tunnel-through-iap'
alias gcp-scp='gcloud compute scp --tunnel-through-iap --zone=$GCP_ZONE'
```

Run `source ~/.bashrc` after adding. Usage:

```bash
# SSH into the VM as ubuntu
gcp-ssh

# Copy bootstrap script to the VM
gcp-scp cloud/bootstrap_gcp.sh $GCP_USER@$GCP_VM:~/bootstrap_gcp.sh

# Copy any file from VM to local
gcp-scp $GCP_USER@$GCP_VM:~/some_file.txt ./local_copy.txt
```

---

## 7. Install Docker

The Deep Learning VM ships with the NVIDIA Container Toolkit preinstalled, but Docker itself may need to be installed.

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker

# Allow running docker without 'sudo' (requires a re-login)
sudo usermod -aG docker $USER
```

Restart the Docker daemon so it picks up the NVIDIA runtime:

```bash
sudo systemctl restart docker
```

**Exit and SSH back in** for the docker group membership to take effect.

---

## 8. Verify GPU Access Inside Docker

After reconnecting, run:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

Expected output: an `nvidia-smi` table showing the **L4** GPU, driver version, and CUDA version. If this works, containerized PyTorch training jobs will have GPU access.

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Fix |
|---|---|---|
| `curl` hangs from VM | Cloud NAT not configured for the VM's region | Re-check region match in [Section 3](#3-configure-outbound-access-via-cloud-nat) |
| SSH from local times out | IAP firewall rule missing | Re-run the command in [Section 6.2](#62-create-the-iap-firewall-rule-one-time-setup) |
| `docker` requires `sudo` | Group membership not refreshed | Log out and SSH back in |
| `docker run --gpus all` fails | NVIDIA Container Toolkit not registered with Docker | Re-run `sudo systemctl restart docker` |
| `nvidia-smi` empty inside container | Wrong base image or runtime | Use an NVIDIA CUDA base image (e.g., `nvidia/cuda:12.4.0-base-ubuntu22.04`) |
| `nvidia-smi` fails on host with "couldn't communicate with driver" | Kernel module not loaded | Run `sudo ubuntu-drivers autoinstall && sudo reboot` |

---

## Next Steps

With the VM ready, the H-LeWM training stack can be brought up:

1. Clone the LeWorldModel and HWM repositories.
2. Build a Docker image with the project's PyTorch dependencies.
3. Log in to W&B (`wandb login`) and verify the run shows up in the project dashboard.
4. Start with reproducing the LeWorldModel baseline on Push-T before adding the hierarchical predictor `P^(2)` and action encoder `A_Ψ`.