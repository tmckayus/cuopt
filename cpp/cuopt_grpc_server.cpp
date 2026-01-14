/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file cuopt_grpc_server.cpp
 * @brief Minimal gRPC-based remote solve server (prototype)
 *
 * This is a minimal prototype implementing:
 * - SubmitJob RPC: Submit LP/MIP solve jobs
 * - GetResult RPC: Retrieve completed results
 *
 * The server uses the same worker process model as the legacy server,
 * but exposes a gRPC API instead of custom TCP protocol.
 */

#ifdef CUOPT_ENABLE_GRPC

#include <grpcpp/grpcpp.h>
#include "cuopt_remote.pb.h"
#include "cuopt_remote_service.grpc.pb.h"

#include <cuopt/linear_programming/solve.hpp>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

using namespace cuopt::remote;
using namespace cuopt::linear_programming;

// ============================================================================
// Shared Memory Structures (simplified for prototype)
// ============================================================================

constexpr size_t MAX_JOBS = 100;

struct JobQueueEntry {
  char job_id[64];
  uint32_t problem_type;  // 0 = LP, 1 = MIP
  std::atomic<bool> ready;
  std::atomic<bool> claimed;
  std::atomic<pid_t> worker_pid;
  std::atomic<bool> cancelled;
  std::atomic<bool> completed;
};

struct ResultQueueEntry {
  char job_id[64];
  uint32_t status;  // 0 = success, 1 = error
  char error_message[1024];
  std::atomic<bool> ready;
  std::atomic<bool> retrieved;
};

struct SharedMemoryControl {
  std::atomic<bool> shutdown_requested;
  std::atomic<int> active_workers;
  JobQueueEntry job_queue[MAX_JOBS];
  ResultQueueEntry result_queue[MAX_JOBS];
};

// ============================================================================
// Global State
// ============================================================================

static SharedMemoryControl* g_shm_ctrl = nullptr;
static std::mutex g_job_mutex;
static std::map<std::string, int> g_job_index_map;  // job_id -> queue index

// ============================================================================
// Helper Functions
// ============================================================================

std::string generate_job_id()
{
  static std::random_device rd;
  static std::mt19937_64 gen(rd());
  static std::uniform_int_distribution<uint64_t> dis;

  std::stringstream ss;
  ss << std::hex << dis(gen);
  return ss.str();
}

// Simulate worker process (simplified for prototype)
// In real implementation, this would fork and solve the problem
void simulate_worker(int job_idx, const std::string& job_id)
{
  std::this_thread::sleep_for(std::chrono::seconds(2));  // Simulate solve

  // Mark result as ready
  auto& result = g_shm_ctrl->result_queue[job_idx];
  strncpy(result.job_id, job_id.c_str(), sizeof(result.job_id) - 1);
  result.status = 0;  // Success
  strncpy(result.error_message, "Simulated solve completed", sizeof(result.error_message) - 1);
  result.ready.store(true);

  std::cout << "[Worker] Completed job: " << job_id << std::endl;
}

// ============================================================================
// gRPC Service Implementation
// ============================================================================

class CuOptRemoteServiceImpl final : public CuOptRemoteService::Service {
 public:
  Status SubmitJob(ServerContext* context,
                   const SubmitJobRequest* request,
                   SubmitJobResponse* response) override
  {
    std::cout << "[gRPC] SubmitJob called" << std::endl;

    // Generate job ID
    std::string job_id = generate_job_id();
    response->set_job_id(job_id);
    response->set_message("Job submitted successfully");

    // Find free slot in job queue
    std::lock_guard<std::mutex> lock(g_job_mutex);
    int job_idx = -1;
    for (size_t i = 0; i < MAX_JOBS; i++) {
      if (!g_shm_ctrl->job_queue[i].ready.load() && !g_shm_ctrl->job_queue[i].claimed.load()) {
        job_idx = i;
        break;
      }
    }

    if (job_idx == -1) { return Status(StatusCode::RESOURCE_EXHAUSTED, "Job queue full"); }

    // Store job in queue
    auto& job = g_shm_ctrl->job_queue[job_idx];
    strncpy(job.job_id, job_id.c_str(), sizeof(job.job_id) - 1);

    if (request->has_lp_request()) {
      job.problem_type = 0;  // LP
      std::cout << "[gRPC] LP problem submitted" << std::endl;
    } else if (request->has_mip_request()) {
      job.problem_type = 1;  // MIP
      std::cout << "[gRPC] MIP problem submitted" << std::endl;
    } else {
      return Status(StatusCode::INVALID_ARGUMENT, "No problem data provided");
    }

    job.ready.store(true);
    job.claimed.store(false);
    job.completed.store(false);

    g_job_index_map[job_id] = job_idx;

    // Launch worker thread (simplified for prototype)
    std::thread worker_thread(simulate_worker, job_idx, job_id);
    worker_thread.detach();

    std::cout << "[gRPC] Job submitted: " << job_id << " at index " << job_idx << std::endl;
    return Status::OK;
  }

  Status GetResult(ServerContext* context,
                   const GetResultRequest* request,
                   ResultResponse* response) override
  {
    std::string job_id = request->job_id();
    std::cout << "[gRPC] GetResult called for job: " << job_id << std::endl;

    std::lock_guard<std::mutex> lock(g_job_mutex);

    // Find job in map
    auto it = g_job_index_map.find(job_id);
    if (it == g_job_index_map.end()) { return Status(StatusCode::NOT_FOUND, "Job not found"); }

    int job_idx          = it->second;
    auto& result         = g_shm_ctrl->result_queue[job_idx];
    bool result_is_ready = result.ready.load();

    if (!result_is_ready) {
      // Result not ready yet
      return Status(StatusCode::UNAVAILABLE, "Result not ready yet");
    }

    // Return result
    if (result.status == 0) {
      // Success - create dummy solution
      auto* lp_solution = response->mutable_lp_solution();
      lp_solution->set_termination_status(PDLP_OPTIMAL);
      lp_solution->set_primal_objective(42.0);  // Dummy value
      response->set_error_message("");
    } else {
      // Error
      response->set_error_message(result.error_message);
    }

    // Mark as retrieved
    result.retrieved.store(true);

    std::cout << "[gRPC] Result returned for job: " << job_id << std::endl;
    return Status::OK;
  }

  Status CheckStatus(ServerContext* context,
                     const StatusRequest* request,
                     StatusResponse* response) override
  {
    std::string job_id = request->job_id();

    std::lock_guard<std::mutex> lock(g_job_mutex);
    auto it = g_job_index_map.find(job_id);
    if (it == g_job_index_map.end()) {
      response->set_job_status(NOT_FOUND);
      response->set_message("Job not found");
      return Status::OK;
    }

    int job_idx  = it->second;
    auto& job    = g_shm_ctrl->job_queue[job_idx];
    auto& result = g_shm_ctrl->result_queue[job_idx];

    if (result.ready.load()) {
      response->set_job_status(COMPLETED);
      response->set_message("Job completed");
    } else if (job.claimed.load()) {
      response->set_job_status(PROCESSING);
      response->set_message("Job processing");
    } else {
      response->set_job_status(QUEUED);
      response->set_message("Job queued");
    }

    return Status::OK;
  }

  // Stub implementations for other RPCs (not implemented in prototype)
  Status DeleteResult(ServerContext* context,
                      const DeleteRequest* request,
                      DeleteResponse* response) override
  {
    response->set_message("DeleteResult not implemented in prototype");
    return Status::OK;
  }

  Status CancelJob(ServerContext* context,
                   const CancelRequest* request,
                   CancelResponse* response) override
  {
    return Status(StatusCode::UNIMPLEMENTED, "CancelJob not implemented in prototype");
  }

  Status WaitForResult(ServerContext* context,
                       const WaitRequest* request,
                       ResultResponse* response) override
  {
    return Status(StatusCode::UNIMPLEMENTED, "WaitForResult not implemented in prototype");
  }

  Status StreamLogs(ServerContext* context,
                    const StreamLogsRequest* request,
                    grpc::ServerWriter<LogMessage>* writer) override
  {
    return Status(StatusCode::UNIMPLEMENTED, "StreamLogs not implemented in prototype");
  }

  Status SolveSync(ServerContext* context,
                   const SolveSyncRequest* request,
                   SolveSyncResponse* response) override
  {
    return Status(StatusCode::UNIMPLEMENTED, "SolveSync not implemented in prototype");
  }
};

// ============================================================================
// Server Startup
// ============================================================================

void RunServer(const std::string& server_address)
{
  // Initialize shared memory
  g_shm_ctrl = static_cast<SharedMemoryControl*>(mmap(nullptr,
                                                      sizeof(SharedMemoryControl),
                                                      PROT_READ | PROT_WRITE,
                                                      MAP_SHARED | MAP_ANONYMOUS,
                                                      -1,
                                                      0));

  if (g_shm_ctrl == MAP_FAILED) {
    std::cerr << "Failed to create shared memory" << std::endl;
    exit(1);
  }

  // Initialize control block
  new (g_shm_ctrl) SharedMemoryControl();
  g_shm_ctrl->shutdown_requested.store(false);
  g_shm_ctrl->active_workers.store(0);

  // Initialize queues
  for (size_t i = 0; i < MAX_JOBS; i++) {
    g_shm_ctrl->job_queue[i].ready.store(false);
    g_shm_ctrl->job_queue[i].claimed.store(false);
    g_shm_ctrl->job_queue[i].completed.store(false);
    g_shm_ctrl->result_queue[i].ready.store(false);
    g_shm_ctrl->result_queue[i].retrieved.store(false);
  }

  // Create gRPC service
  CuOptRemoteServiceImpl service;

  // Build server
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "[gRPC Server] Listening on " << server_address << std::endl;
  std::cout << "[gRPC Server] Prototype implementation - SubmitJob and GetResult only" << std::endl;

  // Wait for server to shutdown
  server->Wait();

  // Cleanup
  munmap(g_shm_ctrl, sizeof(SharedMemoryControl));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
  std::string server_address = "0.0.0.0:8765";

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-p" || arg == "--port") {
      if (i + 1 < argc) {
        int port       = std::stoi(argv[++i]);
        server_address = "0.0.0.0:" + std::to_string(port);
      }
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  -p, --port PORT    Listen port (default: 8765)\n"
                << "  -h, --help         Show this help\n";
      return 0;
    }
  }

  std::cout << "cuOpt gRPC Server (Prototype)\n"
            << "==============================\n"
            << "Starting server on " << server_address << "\n"
            << std::endl;

  RunServer(server_address);

  return 0;
}

#else  // !CUOPT_ENABLE_GRPC

#include <iostream>

int main()
{
  std::cerr << "Error: cuopt_grpc_server requires gRPC support.\n"
            << "Rebuild with gRPC enabled (CUOPT_ENABLE_GRPC=ON)" << std::endl;
  return 1;
}

#endif  // CUOPT_ENABLE_GRPC
