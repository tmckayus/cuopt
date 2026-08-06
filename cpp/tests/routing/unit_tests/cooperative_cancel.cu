/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <routing/utilities/test_utilities.hpp>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace cuopt::routing::test {

// Homberger R1_10_1: 1000 customers + depot. Under a multi-minute time limit the
// metaheuristic stays busy long enough for a mid-solve cancel to land.
TEST(routing_cooperative_cancel, solve_exits_early_on_cancel_flag)
{
  constexpr float long_time_limit = 300.f;

  raft::handle_t handle;
  auto stream = handle.get_stream();

  const std::string path =
    cuopt::test::get_rapids_dataset_root_dir() + "/cvrptw/R1_4_1.TXT";
  Route<int, float> route;
  // Homberger R1_4_1: 400 customers + depot — long enough to stay busy under a
  // 300s limit, smaller than R1_10_1 so cancel can unwind in a unit-test budget.
  load_cvrptw(path, route, 401);
  ASSERT_GT(route.n_locations, 100) << "failed to load " << path;

  const int nodes      = route.n_locations;
  const int n_orders   = nodes - 1;
  const int n_vehicles = route.n_vehicles;

  std::vector<float> cost_matrix_h(nodes * nodes);
  build_dense_matrix(cost_matrix_h.data(), route.x_h, route.y_h);

  std::vector<int> order_locations_h(n_orders), earliest_h(n_orders), latest_h(n_orders),
    service_h(n_orders), demand_h(n_orders);
  for (int i = 0; i < n_orders; ++i) {
    order_locations_h[i] = i + 1;
    earliest_h[i]        = route.earliest_time_h[i + 1];
    latest_h[i]          = route.latest_time_h[i + 1];
    service_h[i]         = route.service_time_h[i + 1];
    demand_h[i]          = route.demand_h[i + 1];
  }
  std::vector<int> vehicle_start_h(n_vehicles, 0), vehicle_return_h(n_vehicles, 0);

  auto v_cost_matrix     = device_copy(cost_matrix_h, stream);
  auto v_time_matrix     = device_copy(cost_matrix_h, stream);
  auto v_order_locations = device_copy(order_locations_h, stream);
  auto v_earliest        = device_copy(earliest_h, stream);
  auto v_latest          = device_copy(latest_h, stream);
  auto v_service         = device_copy(service_h, stream);
  auto v_demand          = device_copy(demand_h, stream);
  auto v_capacity        = device_copy(route.capacity_h, stream);
  auto v_start           = device_copy(vehicle_start_h, stream);
  auto v_return          = device_copy(vehicle_return_h, stream);

  data_model_view_t<int, float> data_model(&handle, nodes, n_vehicles, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_vehicle_locations(v_start.data(), v_return.data());
  data_model.set_order_time_windows(v_earliest.data(), v_latest.data());
  data_model.set_order_service_times(v_service.data());
  data_model.add_capacity_dimension("weight", v_demand.data(), v_capacity.data());

  // Pre-set cancel: should not burn the 300s budget.
  {
    solver_settings_t<int, float> settings;
    settings.set_time_limit(long_time_limit);
    std::atomic<bool> cancel{true};
    settings.cancel_requested = &cancel;

    auto t0     = std::chrono::steady_clock::now();
    auto result = solve(data_model, settings);
    handle.sync_stream();
    auto secs =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0)
        .count();

    EXPECT_LT(secs, 60) << "pre-set cancel should exit early";
    EXPECT_EQ(result.get_status(), solution_status_t::CANCELLED) << result.get_status_string();
  }

  // Mid-solve cancel while Homberger search is still running under a 300s limit.
  {
    solver_settings_t<int, float> settings;
    settings.set_time_limit(long_time_limit);
    std::atomic<bool> cancel{false};
    settings.cancel_requested = &cancel;

    assignment_t<int> result(solution_status_t::EMPTY, stream);
    auto t0 = std::chrono::steady_clock::now();
    std::thread worker([&] {
      result = solve(data_model, settings);
      handle.sync_stream();
    });

    std::this_thread::sleep_for(std::chrono::seconds(3));
    cancel.store(true, std::memory_order_release);
    worker.join();
    auto secs =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0)
        .count();

    EXPECT_GE(secs, 3) << "solve should still have been running when cancel was set";
    EXPECT_LT(secs, 60) << "cancel should exit well before the 300s time limit";
    EXPECT_EQ(result.get_status(), solution_status_t::CANCELLED) << result.get_status_string();
  }
}

}  // namespace cuopt::routing::test
