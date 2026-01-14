# gRPC-based Remote Solve Architecture

## Overview

This document describes the gRPC-based architecture for cuOpt remote solve, which replaces the custom Protobuf serialization with industry-standard gRPC.

## Benefits of gRPC

1. **Robust and Standard**: Uses HTTP/2, built-in error handling, flow control
2. **Type-safe**: Generated code from .proto files using protoc compiler
3. **Streaming Support**: Native server-side streaming for logs
4. **Better Tooling**: Standard debugging tools, interceptors, middleware
5. **Less Error-Prone**: No custom message framing or serialization code

## Service Definition

### RPC Methods

```protobuf
service CuOptRemoteService {
  // Submit a new LP/MIP solve job (async)
  rpc SubmitJob(SubmitJobRequest) returns (SubmitJobResponse);

  // Check job status
  rpc CheckStatus(StatusRequest) returns (StatusResponse);

  // Get completed result
  rpc GetResult(GetResultRequest) returns (ResultResponse);

  // Delete result from server memory
  rpc DeleteResult(DeleteRequest) returns (DeleteResponse);

  // Stream logs in real-time (server-side streaming)
  rpc StreamLogs(StreamLogsRequest) returns (stream LogMessage);

  // Cancel a queued or running job
  rpc CancelJob(CancelRequest) returns (CancelResponse);

  // Wait for result (blocking call, returns when job completes)
  rpc WaitForResult(WaitRequest) returns (ResultResponse);

  // Synchronous solve (blocking, returns result immediately)
  rpc SolveSync(SolveSyncRequest) returns (SolveSyncResponse);
}
```

### Key Improvements

1. **Log Streaming**: Replace polling with gRPC server-side streaming
   - Client opens stream, server pushes log lines as they arrive
   - More efficient, real-time, less network overhead

2. **Type Safety**: Each RPC has specific request/response types
   - No more generic `AsyncRequest` wrapper with `oneof`
   - Better compile-time checking

3. **Error Handling**: gRPC status codes instead of custom `ResponseStatus`
   - Standard codes: OK, CANCELLED, NOT_FOUND, DEADLINE_EXCEEDED, etc.

4. **Streaming Cancellation**: Built-in support for cancelling streams

## Architecture Components

### 1. gRPC Server (`cuopt_grpc_server`)

- Listens on TCP port (e.g., 8765)
- Implements `CuOptRemoteService` interface
- Manages worker processes (same as current implementation)
- Each worker still uses pipes/shared memory for IPC with main process
- Handles concurrent gRPC requests from multiple clients

### 2. gRPC Client Wrapper (C++)

```cpp
class CuOptGrpcClient {
public:
  CuOptGrpcClient(const std::string& server_address);

  // Async API
  std::string submit_job(const SolveLPRequest& request);
  JobStatus check_status(const std::string& job_id);
  LPSolution get_result(const std::string& job_id);
  void delete_result(const std::string& job_id);
  void cancel_job(const std::string& job_id);

  // Blocking API
  LPSolution wait_for_result(const std::string& job_id);
  LPSolution solve_sync(const SolveLPRequest& request);

  // Log streaming
  void stream_logs(const std::string& job_id,
                   std::function<void(const std::string&)> callback);

private:
  std::unique_ptr<CuOptRemoteService::Stub> stub_;
};
```

### 3. Python gRPC Client Wrapper

```python
class CuOptGrpcClient:
    def __init__(self, server_address: str):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = cuopt_remote_pb2_grpc.CuOptRemoteServiceStub(self.channel)

    def submit_job(self, problem, settings) -> str:
        """Submit job, returns job_id"""

    def check_status(self, job_id: str) -> JobStatus:
        """Check job status"""

    def get_result(self, job_id: str):
        """Get completed result"""

    def stream_logs(self, job_id: str):
        """Generator that yields log lines as they arrive"""
        for log_msg in self.stub.StreamLogs(request):
            yield log_msg.line
```

### 4. Pluggable Architecture

Instead of pluggable serialization, we have pluggable client/server implementations:

```cpp
// Abstract remote client interface
class IRemoteClient {
public:
  virtual ~IRemoteClient() = default;

  virtual std::string submit_job(const ProblemData& problem) = 0;
  virtual JobStatus check_status(const std::string& job_id) = 0;
  virtual Solution get_result(const std::string& job_id) = 0;
  virtual void stream_logs(const std::string& job_id, LogCallback callback) = 0;
  // ... other methods
};

// Implementations:
// - CuOptGrpcClient (gRPC-based)
// - CuOptLegacyClient (current pipe/socket-based)
// - CuOptMockClient (for testing)
```

On the server side:

```cpp
// Abstract remote server interface
class IRemoteServer {
public:
  virtual ~IRemoteServer() = default;

  virtual void start(int port, int num_workers) = 0;
  virtual void stop() = 0;
  virtual void wait() = 0;
};

// Implementations:
// - CuOptGrpcServer (gRPC-based)
// - CuOptLegacyServer (current implementation)
```

## Worker Communication

Two options for worker processes:

### Option 1: Keep Pipes (Current)
- gRPC server receives requests over network
- Server process communicates with workers via pipes (current implementation)
- **Pros**: Minimal changes to worker code
- **Cons**: Still have custom pipe serialization internally

### Option 2: Workers as gRPC Clients
- Workers listen on localhost ports (e.g., 8766, 8767, ...)
- Main process sends jobs to workers via gRPC
- **Pros**: Full gRPC stack, no custom serialization
- **Cons**: More refactoring, workers need to accept connections

**Recommendation**: Start with Option 1 (keep pipes for workers), can migrate to Option 2 later.

## Implementation Phases

### Phase 1: Setup (Current)
- [x] Analyze current protocol
- [ ] Create grpc-implementation branch
- [ ] Add gRPC dependencies (grpc++, protobuf)
- [ ] Create gRPC service definition (.proto)

### Phase 2: Server Implementation
- [ ] Generate gRPC code with protoc
- [ ] Implement CuOptGrpcServer class
- [ ] Implement all RPC methods
- [ ] Keep existing worker/pipe communication
- [ ] Add log streaming support

### Phase 3: C++ Client
- [ ] Implement CuOptGrpcClient wrapper
- [ ] Add to cuopt_cli for testing
- [ ] Test all operations

### Phase 4: Python Client
- [ ] Generate Python gRPC code
- [ ] Implement Python client wrapper
- [ ] Update test scripts
- [ ] Test async operations and log streaming

### Phase 5: Testing & Performance
- [ ] Functional testing (all operations)
- [ ] Performance comparison vs pipe-based
- [ ] Load testing (multiple concurrent clients)
- [ ] Documentation

## Performance Considerations

1. **Message Size**: gRPC handles large messages well (better than raw TCP)
2. **Latency**: HTTP/2 multiplexing may add slight overhead, but negligible for solve times
3. **Throughput**: gRPC is highly optimized, should match or exceed current implementation
4. **Streaming**: Server-side streaming for logs is more efficient than polling

## Migration Path

1. **Dual Implementation**: Keep both gRPC and legacy implementations
2. **Environment Variable**: `CUOPT_REMOTE_PROTOCOL=grpc` or `legacy`
3. **Default**: Start with legacy as default, switch to gRPC after validation
4. **Deprecation**: Remove legacy after performance validation

## Security Considerations

1. **TLS/SSL**: gRPC has built-in TLS support (can enable later)
2. **Authentication**: Can add token-based auth via gRPC metadata
3. **Network Isolation**: Can bind to localhost only for local-only access
4. **Input Validation**: gRPC handles message validation automatically

## Dependencies

- **grpc++**: C++ gRPC library
- **protobuf**: Already have this
- **grpcio**: Python gRPC library (for Python clients)
- **grpcio-tools**: For Python code generation
