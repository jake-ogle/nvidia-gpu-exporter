#include <atomic>
#include <chrono>
#include <csignal>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <nvml.h>
#include <prometheus/collectable.h>
#include <prometheus/exposer.h>
#include <prometheus/metric_family.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> running{true};
void stop(int) { running.store(false); }
class Nvml {
  void *library_;

public:
  Nvml() : library_(dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL)) {
    if (!library_)
      throw std::runtime_error("Cannot load libnvidia-ml.so.1; install NVIDIA "
                               "driver and expose its libraries");
  }
  ~Nvml() { dlclose(library_); }
  template <class F> F symbol(const char *name) {
    auto ptr = dlsym(library_, name);
    if (!ptr)
      throw std::runtime_error(std::string("Missing NVML symbol: ") + name);
    return reinterpret_cast<F>(ptr);
  }
};
class Collector final : public prometheus::Collectable {
  mutable std::mutex mutex_;
  std::vector<prometheus::MetricFamily> snapshot_;

public:
  std::vector<prometheus::MetricFamily> Collect() const override {
    std::lock_guard lock(mutex_);
    return snapshot_;
  }
  void sample(Nvml &nvml) {
    std::vector<prometheus::MetricFamily> next;
    auto gauge = [&](const std::string &name, const std::string &help,
                     double value, const std::string &uuid = "") {
      prometheus::MetricFamily family;
      family.name = "nvidia_gpu_" + name;
      family.help = help;
      family.type = prometheus::MetricType::Gauge;
      prometheus::ClientMetric metric;
      metric.gauge.value = value;
      if (!uuid.empty())
        metric.label.push_back({"uuid", uuid});
      family.metric.push_back(metric);
      for (auto &existing : next)
        if (existing.name == family.name) {
          existing.metric.push_back(metric);
          return;
        }
      next.push_back(std::move(family));
    };
    unsigned count = 0;
    bool ok = nvml.symbol<decltype(&nvmlDeviceGetCount_v2)>(
                  "nvmlDeviceGetCount_v2")(&count) == NVML_SUCCESS;
    for (unsigned i = 0; ok && i < count; ++i) {
      nvmlDevice_t device{};
      if (nvml.symbol<decltype(&nvmlDeviceGetHandleByIndex_v2)>(
              "nvmlDeviceGetHandleByIndex_v2")(i, &device) != NVML_SUCCESS) {
        ok = false;
        break;
      }
      char uuid[NVML_DEVICE_UUID_BUFFER_SIZE]{};
      if (nvml.symbol<decltype(&nvmlDeviceGetUUID)>("nvmlDeviceGetUUID")(
              device, uuid, sizeof uuid) != NVML_SUCCESS) {
        ok = false;
        break;
      }
      nvmlUtilization_t utilization{};
      if (nvml.symbol<decltype(&nvmlDeviceGetUtilizationRates)>(
              "nvmlDeviceGetUtilizationRates")(device, &utilization) ==
          NVML_SUCCESS)
        gauge("utilization_ratio", "GPU utilization ratio",
              utilization.gpu / 100.0, uuid);
      nvmlMemory_t memory{};
      if (nvml.symbol<decltype(&nvmlDeviceGetMemoryInfo)>(
              "nvmlDeviceGetMemoryInfo")(device, &memory) == NVML_SUCCESS) {
        gauge("memory_used_bytes", "Used framebuffer bytes",
              static_cast<double>(memory.used), uuid);
        gauge("memory_total_bytes", "Total framebuffer bytes",
              static_cast<double>(memory.total), uuid);
      }
      unsigned value = 0;
      if (nvml.symbol<decltype(&nvmlDeviceGetTemperature)>(
              "nvmlDeviceGetTemperature")(device, NVML_TEMPERATURE_GPU,
                                          &value) == NVML_SUCCESS)
        gauge("temperature_celsius", "GPU temperature", value, uuid);
      if (nvml.symbol<decltype(&nvmlDeviceGetPowerUsage)>(
              "nvmlDeviceGetPowerUsage")(device, &value) == NVML_SUCCESS)
        gauge("power_watts", "GPU power usage", value / 1000.0, uuid);
    }
    gauge("collector_success", "Whether inventory collection succeeded",
          ok ? 1 : 0);
    std::lock_guard lock(mutex_);
    snapshot_ = std::move(next);
  }
};
} // namespace
int main(int argc, char **argv) {
  try {
    std::string address = "0.0.0.0:9835";
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "Usage: nvidia-gpu-exporter [--listen ADDRESS:PORT]\n";
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--listen")
      address = argv[2];
    else if (argc != 1)
      throw std::runtime_error("Invalid arguments; use --help");
    Nvml nvml;
    auto init = nvml.symbol<decltype(&nvmlInit_v2)>("nvmlInit_v2");
    auto shutdown = nvml.symbol<decltype(&nvmlShutdown)>("nvmlShutdown");
    if (init() != NVML_SUCCESS)
      throw std::runtime_error(
          "NVML initialization failed; check GPU access and driver");
    struct Shutdown {
      decltype(shutdown) function;
      ~Shutdown() { function(); }
    } cleanup{shutdown};
    auto collector = std::make_shared<Collector>();
    collector->sample(nvml);
    prometheus::Exposer exposer{address};
    exposer.RegisterCollectable(collector);
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    while (running.load()) {
      for (int i = 0; i < 50 && running.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (running.load())
        collector->sample(nvml);
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
