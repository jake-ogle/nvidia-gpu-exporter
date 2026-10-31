# NVIDIA GPU exporter

C++20 per-node Prometheus exporter. This initial implementation collects physical GPU utilization, framebuffer usage, temperature, and power through dynamically loaded NVML. Unsupported measurements are omitted. Collection runs every five seconds independently of HTTP scrapes.

## Build (Linux)

Install CMake >=3.20, a C++20 compiler, Git, and NVIDIA development headers providing `nvml.h`. The NVIDIA driver supplies `libnvidia-ml.so.1` at runtime. CMake fetches prometheus-cpp v1.3.0.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
sudo cmake --install build
nvidia-gpu-exporter --listen 0.0.0.0:9835
curl localhost:9835/metrics
```

Run one service on each GPU host. Copy `deploy/nvidia-gpu-exporter.service` to the systemd unit directory and enable it. Configure Prometheus with each host as a target:

```yaml
scrape_configs:
  - job_name: nvidia-gpu
    static_configs:
      - targets: ['gpu-node-1:9835', 'gpu-node-2:9835']
```

GPU series use UUID labels; Prometheus adds instance labels. Bind to a private interface or place the endpoint behind an authenticated TLS proxy.

## Implementation status

This is an initial NVML implementation, not the complete planned release. DCGM, MIG/profiling, Kubernetes attribution and deployment, health endpoints, configurable sampling, collector recovery, expanded telemetry, and automated tests remain to be implemented. Collection failures can yield partial snapshots; collector_success currently reflects inventory success only. Hardware and compilation validation have not been performed in this macOS workspace.
