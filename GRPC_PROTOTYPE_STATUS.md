# gRPC Prototype Status

## ✅ Completed (Commits: 4b86e404, bcf6c36d)

### Phase 1: Infrastructure
- **gRPC Service Definition** (`cuopt_remote_service.proto`)
  - 8 RPC methods defined
  - Server-side streaming for logs
  - Type-safe request/response messages

- **Build System Integration**
  - Added gRPC dependencies (grpc-cpp, libgrpc)
  - CMake rules to generate C++ code with `protoc` and `grpc_cpp_plugin`
  - Optional compilation with `CUOPT_ENABLE_GRPC` flag

- **Architecture Documentation** (`GRPC_ARCHITECTURE.md`)
  - Complete design document
  - Migration path from legacy to gRPC
  - Pluggable client/server architecture

### Phase 2: Minimal Prototype
- **gRPC Server** (`cuopt_grpc_server.cpp`)
  - ✅ `SubmitJob` RPC - Submit LP/MIP solve jobs
  - ✅ `GetResult` RPC - Retrieve completed results
  - ✅ `CheckStatus` RPC - Poll for job status
  - ⏸️ `DeleteResult` RPC - Stub (returns OK)
  - ❌ `CancelJob` RPC - Not implemented
  - ❌ `WaitForResult` RPC - Not implemented
  - ❌ `StreamLogs` RPC - Not implemented
  - ❌ `SolveSync` RPC - Not implemented

- **Test Client** (`test_grpc_client.cpp`)
  - Submits simple LP problem
  - Polls for completion
  - Retrieves result
  - Full end-to-end test flow

- **Current Limitations**
  - **Worker simulation**: Uses mock worker threads instead of real solve
  - **No actual solving**: Just sleeps for 2 seconds and returns dummy result
  - **No pipe/IPC**: Doesn't communicate with real worker processes yet
  - **Memory only**: No shared memory for problem data transfer

## 🔄 Next Steps

### To Complete Minimal Prototype
1. **Integrate Real Solver**
   - Replace `simulate_worker()` with actual fork + solve
   - Use shared memory for problem data transfer
   - Implement worker process communication

2. **Test End-to-End**
   - Build with gRPC enabled
   - Run server and client
   - Verify actual LP/MIP solves work

3. **Add Remaining RPCs** (for full implementation)
   - `DeleteResult` - Free memory
   - `CancelJob` - Cancel running jobs
   - `WaitForResult` - Blocking wait
   - `StreamLogs` - Server-side streaming
   - `SolveSync` - Synchronous solve

### To Compare with Legacy
1. **Performance Testing**
   - Same problem on both servers
   - Measure latency and throughput
   - Test with multiple concurrent clients

2. **Feature Parity**
   - Ensure all legacy features work with gRPC
   - Async workflow
   - Log streaming
   - Job cancellation

## 📦 Files Created

### Core gRPC Files
- `cpp/src/linear_programming/utilities/cuopt_remote_service.proto` - gRPC service definition
- `cpp/cuopt_grpc_server.cpp` - gRPC server implementation
- `cpp/test_grpc_client.cpp` - Test client
- `GRPC_ARCHITECTURE.md` - Architecture documentation
- `GRPC_PROTOTYPE_STATUS.md` - This file

### Modified Files
- `dependencies.yaml` - Added gRPC dependencies
- `cpp/CMakeLists.txt` - Build rules for gRPC
- `conda/environments/*.yaml` - Regenerated with gRPC

## 🗑️ Files to Delete (After Legacy Deprecation)

When we fully migrate to gRPC and deprecate the legacy server, these files can be removed:

### Custom Serialization Infrastructure
- `cpp/include/cuopt/linear_programming/utilities/remote_serialization.hpp` - Pluggable serializer interface
- `cpp/src/linear_programming/utilities/protobuf_serializer.cu` - Custom protobuf serializer
- `cpp/src/linear_programming/utilities/serializers/` - Plugin directory
  - `msgpack_serializer.cpp` - Msgpack plugin (can delete now, not used)
  - `CMakeLists.txt` - Plugin build file

### Legacy Server (Keep for Now)
- `cpp/cuopt_remote_server.cpp` - Current TCP-based server (keep for comparison)

**Rationale**: gRPC handles all serialization internally with generated code from `.proto` files. The custom pluggable serialization framework (`remote_serialization.hpp`, `protobuf_serializer.cu`) was necessary for the legacy TCP-based protocol but is redundant with gRPC.

## 🧪 Testing the Prototype

### Prerequisites
```bash
# Ensure gRPC is available in conda environment
conda install grpc-cpp libgrpc -c conda-forge

# Build with gRPC enabled
cd /home/tmckay/repos/nvidia-cuopt
./build.sh  # or your build command with -DBUILD_REMOTE_SERVER=ON
```

### Run Server
```bash
cd cpp/build
./cuopt_grpc_server -p 8765
```

### Run Test Client
```bash
cd cpp/build
./test_grpc_client localhost:8765
```

Expected output:
```
[Client] Job submitted successfully
[Client] Job ID: <hex-id>
[Client] Polling for job completion...
[Client] Job status: QUEUED - Job queued
[Client] Job status: PROCESSING - Job processing
[Client] Job status: COMPLETED - Job completed
[Client] Retrieving result...
[Client] Result retrieved successfully
[Client] LP Solution:
[Client]   Status: 2  (PDLP_OPTIMAL)
[Client]   Objective: 42.0  (dummy value)
[Client] Test completed successfully!
```

## 📊 Comparison: Legacy vs gRPC

| Feature | Legacy Server | gRPC Server (Prototype) |
|---------|---------------|------------------------|
| Protocol | Custom TCP | gRPC (HTTP/2) |
| Serialization | Custom (protobuf) | gRPC-generated |
| Message Framing | Manual (4-byte length prefix) | Built-in |
| Error Handling | Custom status codes | gRPC Status codes |
| Log Streaming | Polling (GET_LOGS) | Server-side streaming (future) |
| Type Safety | Runtime | Compile-time |
| Tooling | Custom | Industry-standard |
| Code Lines | ~2300 | ~350 (for same features) |

## 🎯 Success Criteria

The gRPC prototype will be considered successful when:
1. ✅ Compiles with gRPC enabled
2. ⏳ Server accepts SubmitJob requests
3. ⏳ Server executes real LP/MIP solves (not just simulation)
4. ⏳ Client retrieves actual results
5. ⏳ Performance is comparable to legacy server
6. ⏳ Code is simpler and more maintainable

**Current Status**: ✅ 1/6 complete (builds successfully)
