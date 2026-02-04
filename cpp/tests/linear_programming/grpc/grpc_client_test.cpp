/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_client_test.cpp
 * @brief Unit tests for grpc_client_t using mock stubs
 *
 * These tests verify client-side error handling without requiring a real server.
 * For integration tests with a real server, see grpc_integration_test.cpp.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "grpc_client_test_helper.hpp"

#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/utilities/grpc_client.hpp>

#include <cuopt_remote_service.grpc.pb.h>
#include <cuopt_remote_service.pb.h>
#include <grpcpp/grpcpp.h>

using namespace cuopt::linear_programming;
using namespace ::testing;

/**
 * @brief Mock stub for CuOptRemoteService
 *
 * This mock allows us to control exactly what the "server" returns
 * without running an actual server.
 */
class MockCuOptStub : public cuopt::remote::CuOptRemoteService::StubInterface {
 public:
  // Unary RPCs
  MOCK_METHOD(grpc::Status,
              SubmitJob,
              (grpc::ClientContext*,
               const cuopt::remote::SubmitJobRequest&,
               cuopt::remote::SubmitJobResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              CheckStatus,
              (grpc::ClientContext*,
               const cuopt::remote::StatusRequest&,
               cuopt::remote::StatusResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              GetResult,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultRequest&,
               cuopt::remote::ResultResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              DeleteResult,
              (grpc::ClientContext*,
               const cuopt::remote::DeleteRequest&,
               cuopt::remote::DeleteResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              CancelJob,
              (grpc::ClientContext*,
               const cuopt::remote::CancelRequest&,
               cuopt::remote::CancelResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              WaitForCompletion,
              (grpc::ClientContext*,
               const cuopt::remote::WaitRequest&,
               cuopt::remote::WaitResponse*),
              (override));

  MOCK_METHOD(grpc::Status,
              GetIncumbents,
              (grpc::ClientContext*,
               const cuopt::remote::IncumbentRequest&,
               cuopt::remote::IncumbentResponse*),
              (override));

  // Streaming RPCs - these need special handling
  MOCK_METHOD(grpc::ClientReaderInterface<cuopt::remote::ResultChunk>*,
              StreamResultRaw,
              (grpc::ClientContext*, const cuopt::remote::GetResultRequest&),
              (override));

  MOCK_METHOD(grpc::ClientReaderInterface<cuopt::remote::LogMessage>*,
              StreamLogsRaw,
              (grpc::ClientContext*, const cuopt::remote::StreamLogsRequest&),
              (override));

  // Bidirectional streaming for upload
  MOCK_METHOD((grpc::ClientReaderWriterInterface<cuopt::remote::UploadJobRequest,
                                                 cuopt::remote::UploadJobResponse>*),
              UploadAndSubmitRaw,
              (grpc::ClientContext*),
              (override));

  // Required by interface - async versions (not used in our client but required for interface)
  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::SubmitJobResponse>*,
              AsyncSubmitJobRaw,
              (grpc::ClientContext*,
               const cuopt::remote::SubmitJobRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::SubmitJobResponse>*,
              PrepareAsyncSubmitJobRaw,
              (grpc::ClientContext*,
               const cuopt::remote::SubmitJobRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::StatusResponse>*,
              AsyncCheckStatusRaw,
              (grpc::ClientContext*, const cuopt::remote::StatusRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::StatusResponse>*,
              PrepareAsyncCheckStatusRaw,
              (grpc::ClientContext*, const cuopt::remote::StatusRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::ResultResponse>*,
              AsyncGetResultRaw,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::ResultResponse>*,
              PrepareAsyncGetResultRaw,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::DeleteResponse>*,
              AsyncDeleteResultRaw,
              (grpc::ClientContext*, const cuopt::remote::DeleteRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::DeleteResponse>*,
              PrepareAsyncDeleteResultRaw,
              (grpc::ClientContext*, const cuopt::remote::DeleteRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::CancelResponse>*,
              AsyncCancelJobRaw,
              (grpc::ClientContext*, const cuopt::remote::CancelRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::CancelResponse>*,
              PrepareAsyncCancelJobRaw,
              (grpc::ClientContext*, const cuopt::remote::CancelRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::WaitResponse>*,
              AsyncWaitForCompletionRaw,
              (grpc::ClientContext*, const cuopt::remote::WaitRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::WaitResponse>*,
              PrepareAsyncWaitForCompletionRaw,
              (grpc::ClientContext*, const cuopt::remote::WaitRequest&, grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::IncumbentResponse>*,
              AsyncGetIncumbentsRaw,
              (grpc::ClientContext*,
               const cuopt::remote::IncumbentRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(grpc::ClientAsyncResponseReaderInterface<cuopt::remote::IncumbentResponse>*,
              PrepareAsyncGetIncumbentsRaw,
              (grpc::ClientContext*,
               const cuopt::remote::IncumbentRequest&,
               grpc::CompletionQueue*),
              (override));

  // Async streaming methods
  MOCK_METHOD(
    grpc::ClientAsyncReaderInterface<cuopt::remote::ResultChunk>*,
    AsyncStreamResultRaw,
    (grpc::ClientContext*, const cuopt::remote::GetResultRequest&, grpc::CompletionQueue*, void*),
    (override));

  MOCK_METHOD(grpc::ClientAsyncReaderInterface<cuopt::remote::ResultChunk>*,
              PrepareAsyncStreamResultRaw,
              (grpc::ClientContext*,
               const cuopt::remote::GetResultRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD(
    grpc::ClientAsyncReaderInterface<cuopt::remote::LogMessage>*,
    AsyncStreamLogsRaw,
    (grpc::ClientContext*, const cuopt::remote::StreamLogsRequest&, grpc::CompletionQueue*, void*),
    (override));

  MOCK_METHOD(grpc::ClientAsyncReaderInterface<cuopt::remote::LogMessage>*,
              PrepareAsyncStreamLogsRaw,
              (grpc::ClientContext*,
               const cuopt::remote::StreamLogsRequest&,
               grpc::CompletionQueue*),
              (override));

  MOCK_METHOD((grpc::ClientAsyncReaderWriterInterface<cuopt::remote::UploadJobRequest,
                                                      cuopt::remote::UploadJobResponse>*),
              AsyncUploadAndSubmitRaw,
              (grpc::ClientContext*, grpc::CompletionQueue*, void*),
              (override));

  MOCK_METHOD((grpc::ClientAsyncReaderWriterInterface<cuopt::remote::UploadJobRequest,
                                                      cuopt::remote::UploadJobResponse>*),
              PrepareAsyncUploadAndSubmitRaw,
              (grpc::ClientContext*, grpc::CompletionQueue*),
              (override));
};

/**
 * @brief Test fixture for grpc_client_t tests with mock stub injection
 */
class GrpcClientTest : public ::testing::Test {
 protected:
  std::shared_ptr<MockCuOptStub> mock_stub_;
  std::unique_ptr<grpc_client_t> client_;

  void SetUp() override
  {
    // Create a mock stub
    mock_stub_ = std::make_shared<MockCuOptStub>();

    // Create a client and inject the mock stub
    grpc_client_config_t config;
    config.server_address = "mock://test";
    client_               = std::make_unique<grpc_client_t>(config);

    // Inject the mock stub using typed helper
    grpc_test_inject_mock_stub_typed(*client_, mock_stub_);
  }

  void TearDown() override
  {
    client_.reset();
    mock_stub_.reset();
  }
};

// =============================================================================
// CheckStatus Tests
// =============================================================================

TEST_F(GrpcClientTest, CheckStatus_Success_Completed)
{
  // Setup mock to return COMPLETED status
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest& req,
                 cuopt::remote::StatusResponse* resp) {
      EXPECT_EQ(req.job_id(), "test-job-123");
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_message("Job completed successfully");
      resp->set_result_size_bytes(1024);
      return grpc::Status::OK;
    });

  auto result = client_->check_status("test-job-123");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.status, job_status_t::COMPLETED);
  EXPECT_EQ(result.message, "Job completed successfully");
  EXPECT_EQ(result.result_size_bytes, 1024);
}

TEST_F(GrpcClientTest, CheckStatus_Success_Processing)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::PROCESSING);
      resp->set_message("Solving...");
      return grpc::Status::OK;
    });

  auto result = client_->check_status("test-job-456");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.status, job_status_t::PROCESSING);
}

TEST_F(GrpcClientTest, CheckStatus_JobNotFound)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::NOT_FOUND);
      resp->set_message("Job not found");
      return grpc::Status::OK;
    });

  auto result = client_->check_status("nonexistent-job");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.status, job_status_t::NOT_FOUND);
}

TEST_F(GrpcClientTest, CheckStatus_RpcFailure_Unavailable)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse*) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Server unavailable");
    });

  auto result = client_->check_status("test-job");

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Server unavailable") != std::string::npos);
}

TEST_F(GrpcClientTest, CheckStatus_RpcFailure_Internal)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse*) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "Internal server error");
    });

  auto result = client_->check_status("test-job");

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Internal server error") != std::string::npos);
}

// =============================================================================
// CancelJob Tests
// =============================================================================

TEST_F(GrpcClientTest, CancelJob_Success)
{
  EXPECT_CALL(*mock_stub_, CancelJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::CancelRequest& req,
                 cuopt::remote::CancelResponse* resp) {
      EXPECT_EQ(req.job_id(), "job-to-cancel");
      resp->set_job_status(cuopt::remote::CANCELLED);
      resp->set_message("Job cancelled");
      return grpc::Status::OK;
    });

  auto result = client_->cancel_job("job-to-cancel");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.job_status, job_status_t::CANCELLED);
}

TEST_F(GrpcClientTest, CancelJob_AlreadyCompleted)
{
  EXPECT_CALL(*mock_stub_, CancelJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::CancelRequest&,
                 cuopt::remote::CancelResponse* resp) {
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_message("Job already completed");
      return grpc::Status::OK;
    });

  auto result = client_->cancel_job("completed-job");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.job_status, job_status_t::COMPLETED);
}

TEST_F(GrpcClientTest, CancelJob_RpcFailure)
{
  EXPECT_CALL(*mock_stub_, CancelJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::CancelRequest&,
                 cuopt::remote::CancelResponse*) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Server down");
    });

  auto result = client_->cancel_job("job-id");

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Server down") != std::string::npos);
}

// =============================================================================
// DeleteJob Tests
// =============================================================================

TEST_F(GrpcClientTest, DeleteJob_Success)
{
  EXPECT_CALL(*mock_stub_, DeleteResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::DeleteRequest& req,
                 cuopt::remote::DeleteResponse* resp) {
      EXPECT_EQ(req.job_id(), "job-to-delete");
      resp->set_status(cuopt::remote::SUCCESS);
      return grpc::Status::OK;
    });

  bool result = client_->delete_job("job-to-delete");

  EXPECT_TRUE(result);
}

TEST_F(GrpcClientTest, DeleteJob_NotFound)
{
  EXPECT_CALL(*mock_stub_, DeleteResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::DeleteRequest&,
                 cuopt::remote::DeleteResponse* resp) {
      resp->set_status(cuopt::remote::ERROR_NOT_FOUND);
      return grpc::Status::OK;
    });

  bool result = client_->delete_job("nonexistent-job");

  // Job not found should return false to prevent silent failures
  EXPECT_FALSE(result);
}

TEST_F(GrpcClientTest, DeleteJob_RpcFailure)
{
  EXPECT_CALL(*mock_stub_, DeleteResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::DeleteRequest&,
                 cuopt::remote::DeleteResponse*) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "Delete failed");
    });

  bool result = client_->delete_job("job-id");

  EXPECT_FALSE(result);
}

// =============================================================================
// WaitForCompletion Tests
// =============================================================================

TEST_F(GrpcClientTest, WaitForCompletion_Success)
{
  EXPECT_CALL(*mock_stub_, WaitForCompletion(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::WaitRequest& req,
                 cuopt::remote::WaitResponse* resp) {
      EXPECT_EQ(req.job_id(), "wait-job");
      resp->set_job_status(cuopt::remote::COMPLETED);
      resp->set_message("Done");
      resp->set_result_size_bytes(2048);
      return grpc::Status::OK;
    });

  auto result = client_->wait_for_completion("wait-job");

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.status, job_status_t::COMPLETED);
  EXPECT_EQ(result.result_size_bytes, 2048);
}

TEST_F(GrpcClientTest, WaitForCompletion_Failed)
{
  EXPECT_CALL(*mock_stub_, WaitForCompletion(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::WaitRequest&,
                 cuopt::remote::WaitResponse* resp) {
      resp->set_job_status(cuopt::remote::FAILED);
      resp->set_message("Solve failed: out of memory");
      return grpc::Status::OK;
    });

  auto result = client_->wait_for_completion("failed-job");

  EXPECT_TRUE(result.success);  // RPC succeeded, job failed
  EXPECT_EQ(result.status, job_status_t::FAILED);
  EXPECT_TRUE(result.message.find("out of memory") != std::string::npos);
}

TEST_F(GrpcClientTest, WaitForCompletion_RpcTimeout)
{
  EXPECT_CALL(*mock_stub_, WaitForCompletion(_, _, _))
    .WillOnce(
      [](grpc::ClientContext*, const cuopt::remote::WaitRequest&, cuopt::remote::WaitResponse*) {
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "Deadline exceeded");
      });

  auto result = client_->wait_for_completion("timeout-job");

  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.error_message.find("Deadline exceeded") != std::string::npos);
}

// =============================================================================
// GetIncumbents Tests
// =============================================================================

TEST_F(GrpcClientTest, GetIncumbents_Success)
{
  EXPECT_CALL(*mock_stub_, GetIncumbents(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::IncumbentRequest& req,
                 cuopt::remote::IncumbentResponse* resp) {
      EXPECT_EQ(req.job_id(), "mip-job");
      EXPECT_EQ(req.from_index(), 0);

      auto* inc1 = resp->add_incumbents();
      inc1->set_index(0);
      inc1->set_objective(100.5);
      inc1->add_assignment(1.0);
      inc1->add_assignment(0.0);

      auto* inc2 = resp->add_incumbents();
      inc2->set_index(1);
      inc2->set_objective(95.3);
      inc2->add_assignment(1.0);
      inc2->add_assignment(1.0);

      resp->set_next_index(2);
      resp->set_job_complete(false);
      return grpc::Status::OK;
    });

  auto result = client_->get_incumbents("mip-job", 0, 10);

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.incumbents.size(), 2);
  EXPECT_EQ(result.incumbents[0].index, 0);
  EXPECT_DOUBLE_EQ(result.incumbents[0].objective, 100.5);
  EXPECT_EQ(result.incumbents[1].index, 1);
  EXPECT_DOUBLE_EQ(result.incumbents[1].objective, 95.3);
  EXPECT_EQ(result.next_index, 2);
  EXPECT_FALSE(result.job_complete);
}

TEST_F(GrpcClientTest, GetIncumbents_NoNewIncumbents)
{
  EXPECT_CALL(*mock_stub_, GetIncumbents(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::IncumbentRequest& req,
                 cuopt::remote::IncumbentResponse* resp) {
      resp->set_next_index(req.from_index());  // No new incumbents
      resp->set_job_complete(false);
      return grpc::Status::OK;
    });

  auto result = client_->get_incumbents("mip-job", 5, 10);

  EXPECT_TRUE(result.success);
  EXPECT_TRUE(result.incumbents.empty());
  EXPECT_EQ(result.next_index, 5);
}

// =============================================================================
// Connection Test (without mock - tests real connection failure)
// =============================================================================

TEST(GrpcClientConnectionTest, Connect_ServerUnavailable)
{
  grpc_client_config_t config;
  config.server_address  = "localhost:1";  // Invalid port
  config.timeout_seconds = 1;

  grpc_client_t client(config);
  EXPECT_FALSE(client.connect());
  EXPECT_FALSE(client.get_last_error().empty());
}

TEST(GrpcClientConnectionTest, IsConnected_BeforeConnect)
{
  grpc_client_config_t config;
  config.server_address = "localhost:9999";

  grpc_client_t client(config);
  EXPECT_FALSE(client.is_connected());
}

// =============================================================================
// Transient Failure / Retry Behavior Tests
// =============================================================================

TEST_F(GrpcClientTest, CheckStatus_TransientFailureThenSuccess)
{
  // First call fails with UNAVAILABLE (transient), second succeeds
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse*) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Temporary failure");
    })
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      resp->set_job_status(cuopt::remote::COMPLETED);
      return grpc::Status::OK;
    });

  // First call should fail
  auto result1 = client_->check_status("retry-job");
  EXPECT_FALSE(result1.success);

  // Second call should succeed (simulates retry at higher level)
  auto result2 = client_->check_status("retry-job");
  EXPECT_TRUE(result2.success);
  EXPECT_EQ(result2.status, job_status_t::COMPLETED);
}

TEST_F(GrpcClientTest, GetResult_InternalError)
{
  // Server reports internal error
  EXPECT_CALL(*mock_stub_, GetResult(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::GetResultRequest&,
                 cuopt::remote::ResultResponse*) {
      return grpc::Status(grpc::StatusCode::INTERNAL, "Internal server error");
    });

  auto result = client_->get_lp_result<int32_t, double>("error-job");
  EXPECT_FALSE(result.success);
  EXPECT_FALSE(result.error_message.empty());
}

TEST_F(GrpcClientTest, CancelJob_DeadlineExceeded)
{
  EXPECT_CALL(*mock_stub_, CancelJob(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::CancelRequest&,
                 cuopt::remote::CancelResponse*) {
      return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "Request timeout");
    });

  auto result = client_->cancel_job("timeout-job");
  EXPECT_FALSE(result.success);
}

// =============================================================================
// Malformed Response Tests
// =============================================================================

TEST_F(GrpcClientTest, CheckStatus_MalformedResponse_InvalidStatus)
{
  EXPECT_CALL(*mock_stub_, CheckStatus(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::StatusRequest&,
                 cuopt::remote::StatusResponse* resp) {
      // Set an invalid/unexpected status value
      resp->set_job_status(static_cast<cuopt::remote::JobStatus>(999));
      return grpc::Status::OK;
    });

  auto result = client_->check_status("malformed-job");

  // Should handle gracefully - either map to unknown or report error
  EXPECT_TRUE(result.success);  // RPC succeeded
}

TEST_F(GrpcClientTest, GetIncumbents_MalformedResponse_NegativeIndex)
{
  EXPECT_CALL(*mock_stub_, GetIncumbents(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::IncumbentRequest&,
                 cuopt::remote::IncumbentResponse* resp) {
      auto* inc = resp->add_incumbents();
      inc->set_index(-1);  // Invalid negative index
      inc->set_objective(100.0);
      resp->set_next_index(-5);  // Invalid
      return grpc::Status::OK;
    });

  auto result = client_->get_incumbents("malformed-job", 0, 10);

  // Should handle gracefully
  EXPECT_TRUE(result.success);
}

TEST_F(GrpcClientTest, WaitForCompletion_EmptyMessage)
{
  EXPECT_CALL(*mock_stub_, WaitForCompletion(_, _, _))
    .WillOnce([](grpc::ClientContext*,
                 const cuopt::remote::WaitRequest&,
                 cuopt::remote::WaitResponse* resp) {
      // Don't set any fields - empty response
      return grpc::Status::OK;
    });

  auto result = client_->wait_for_completion("empty-response-job");

  // Should handle gracefully with default values
  EXPECT_TRUE(result.success);
}
