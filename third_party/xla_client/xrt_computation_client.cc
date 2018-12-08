#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"

#include <cstdlib>
#include <functional>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/unique.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/compiler/xla/xla_client/xrt_local_service.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace xla {
namespace {

static const char* const kCpuDevice = "/device:CPU:0";

}  // namespace

XrtComputationClient::XrtComputationClient(
    XrtComputationClient::Options options)
    : options_(std::move(options)) {
  auto default_device_target =
      options_.device_map.find(options_.default_device);
  XLA_CHECK(default_device_target != options_.device_map.end());
  for (const auto& dev_target : options_.device_map) {
    LOG(INFO) << "XRT device " << dev_target.first << " -> "
              << dev_target.second;
  }
  LOG(INFO) << "XRT default device: " << default_device_target->first;
  MaybeCreateLocalService(options_);
  InitializeDevices();
  StartHandleReleaser();
}

void XrtComputationClient::FlushLazyReleases() {
  // Activate the lazy handle releaser and wait for it to complete our run.
  size_t run_id = triggered_task_->Activate();
  triggered_task_->WaitForRun(run_id);
}

std::vector<std::shared_ptr<ComputationClient::Data>>
XrtComputationClient::TransferToServer(
    tensorflow::gtl::ArraySlice<const LiteralDevice> literals) {
  metrics::TimedSection timed(TransferToServerMetric());

  std::mutex lock;
  XrtSessionCache::SessionMap session_map;
  int64 total_size = 0;
  xla_util::MultiWait mwait(literals.size());
  std::map<XrtSession*, SessionWork> session_work_map;
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<Literal> literals_storage(literals.size());
  std::vector<const Literal*> literals_ptrs(literals.size());
  for (size_t i = 0; i < literals.size(); ++i) {
    auto converter = [&, i]() {
      const Literal& literal = literals[i].GetLiteral(&literals_storage[i]);
      literals_ptrs[i] = &literal;

      string device = GetEffectiveDevice(literals[i].device);
      const string& xrt_device = TorchDeviceToXrtDevice(device);
      xrt::XLAAllocation alloc;
      alloc.set_device_ordinal(GetDeviceOrdinal(xrt_device));
      *alloc.mutable_value() = literal.ToProto();
      tensorflow::Input::Initializer feed_value(alloc.SerializeAsString());

      {
        std::lock_guard<std::mutex> slock(lock);
        XrtSession* session = GetSessionForXrtDevice(xrt_device, &session_map);
        tensorflow::Scope device_scope =
            session->root()->WithDevice(xrt_device);
        const XrtSession::CachedNode& cached_node =
            GetAllocateNode(session, device_scope, device);
        feed_inputs.insert({cached_node.holders[0], std::move(feed_value)});
        SessionWork* session_work = &session_work_map[session];
        session_work->outputs_handles.push_back(*cached_node.output);
        session_work->index_mapping.push_back(i);

        total_size += literal.size_bytes();
      }
      mwait.Done();
    };
    xla_env::ScheduleClosure(std::move(converter));
  }
  mwait.Wait();

  OutboundDataMetric()->AddSample(total_size);

  std::vector<std::shared_ptr<Data>> results(literals.size());
  for (auto& session_work : session_work_map) {
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session_work.first->session()->Run(
        feed_inputs, session_work.second.outputs_handles, &outputs));
    XLA_CHECK_EQ(outputs.size(), session_work.second.outputs_handles.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      size_t li = session_work.second.index_mapping[i];
      results[li] = std::make_shared<XrtData>(
          GetEffectiveDevice(literals[li].device), outputs[i].scalar<int64>()(),
          literals_ptrs[li]->shape(),
          [this](XrtData* xrt_data) { ReleaseXrtData(xrt_data); });
    }
  }
  return results;
}

std::vector<Literal> XrtComputationClient::TransferFromServer(
    tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> handles) {
  metrics::TimedSection timed(TransferFromServerMetric());

  XrtSessionCache::SessionMap session_map;
  std::map<XrtSession*, SessionWork> session_work_map;
  tensorflow::ClientSession::FeedType feed_inputs;
  for (size_t i = 0; i < handles.size(); ++i) {
    const XrtData& xrt_data = dynamic_cast<const XrtData&>(*handles[i]);
    XrtSession* session = GetSessionForDevice(xrt_data.device(), &session_map);
    tensorflow::Scope device_scope =
        session->root()->WithDevice(TorchDeviceToXrtDevice(xrt_data.device()));
    const XrtSession::CachedNode& cached_node =
        GetReadNode(session, device_scope, xrt_data.device());
    feed_inputs.insert({cached_node.holders[0], xrt_data.handle});
    SessionWork* session_work = &session_work_map[session];
    session_work->outputs_handles.push_back(*cached_node.output);
    session_work->index_mapping.push_back(i);
  }

  int64 total_size = 0;
  std::vector<Literal> results(handles.size());
  for (auto& session_work : session_work_map) {
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session_work.first->session()->Run(
        feed_inputs, session_work.second.outputs_handles, &outputs));
    XLA_CHECK_EQ(outputs.size(), session_work.second.outputs_handles.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      size_t li = session_work.second.index_mapping[i];
      LiteralProto response;
      XLA_CHECK(response.ParseFromString(outputs[i].scalar<string>()()));
      results[li] = std::move(Literal::CreateFromProto(response).ValueOrDie());
      total_size += results[li].size_bytes();
    }
  }
  InboundDataMetric()->AddSample(total_size);
  return results;
}

std::shared_ptr<ComputationClient::Data>
XrtComputationClient::ExecuteComputation(
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<Data*> arguments, const string& device,
    const Shape* output_shape) {
  metrics::TimedSection timed(ExecuteMetric());

  XrtSessionCache::SessionMap session_map;
  string effective_device = GetEffectiveDevice(device);
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<ExecuteContext> exec_ops = CreateExecuteOps(
      &session_map, computation, BuildParallelArguments(arguments),
      output_shape, {effective_device}, &feed_inputs);

  XrtSession* session = GetSessionForDevice(effective_device, &session_map);
  std::vector<tensorflow::Tensor> outputs;
  xrt_util::CheckComputationStatus(
      session->session()->Run(feed_inputs, {exec_ops.front().execute_output},
                              &outputs),
      {&computation});
  XLA_CHECK_EQ(outputs.size(), 1);

  return std::make_shared<XrtData>(
      effective_device, outputs[0].scalar<int64>()(),
      exec_ops.front().result_shape,
      [this](XrtData* xrt_data) { ReleaseXrtData(xrt_data); });
}

std::vector<std::shared_ptr<ComputationClient::Data>>
XrtComputationClient::ExecuteReplicated(
    const XlaComputation& computation,
    const std::vector<std::vector<Data*>>& arguments,
    tensorflow::gtl::ArraySlice<const string> devices,
    const Shape* output_shape) {
  metrics::TimedSection timed(ExecuteReplicatedMetric());

  XrtSessionCache::SessionMap session_map;
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<ExecuteContext> exec_ops =
      CreateExecuteOps(&session_map, computation, arguments, output_shape,
                       devices, &feed_inputs);
  return RunComputations(session_map, exec_ops, {&computation}, devices,
                         feed_inputs);
}

std::vector<std::shared_ptr<ComputationClient::Data>>
XrtComputationClient::RunComputations(
    const XrtSessionCache::SessionMap& session_map,
    const std::vector<ExecuteContext>& exec_ops,
    tensorflow::gtl::ArraySlice<const XlaComputation* const> computations,
    tensorflow::gtl::ArraySlice<const string> devices,
    const tensorflow::ClientSession::FeedType& feed_inputs) {
  // In the PyTorch/XRT interface we keep a map (options_.workers_map) from a
  // worker+taskno, to the GRPC server which is the entry point for that worker.
  // Since XRT could re-distribute ops internally, if we have N hosts
  // (worker+taskno), we could have all the workers pointing to a single GRPC
  // entry point, or we could have each worker pointing directly to the target
  // host.
  // The advantage of the latter approach, is that we do not bottleneck
  // (especially when feeding inputs) the single GRPC entry point.
  // Using the N:1 approach, the session_replicas below will contain a single
  // session, and all the replica executions will go through it (and distributed
  // by XRT on the service side).
  // Chosing the 1:1 approach (one session per worker), we will have N sessions
  // within the session_replicas map, which we will be executing independently.
  std::map<XrtSession*, std::vector<size_t>> session_replicas;
  for (size_t i = 0; i < devices.size(); ++i) {
    auto worker_hostport = GetWorkerForDevice(GetEffectiveDevice(devices[i]));
    XrtSession* session = session_map.at(worker_hostport.second).get();
    session_replicas[session].push_back(i);
  }
  // TODO(dlibenzi): These could be run in parallel.
  std::vector<std::shared_ptr<Data>> results(devices.size());
  for (auto& sess_replica : session_replicas) {
    std::vector<tensorflow::Output> exec_nodes;
    for (auto replica : sess_replica.second) {
      exec_nodes.push_back(exec_ops[replica].execute_output);
    }
    std::vector<tensorflow::Tensor> outputs;
    xrt_util::CheckComputationStatus(
        sess_replica.first->session()->Run(feed_inputs, exec_nodes, &outputs),
        computations);
    XLA_CHECK_EQ(outputs.size(), exec_nodes.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      auto replica = sess_replica.second[i];
      results[replica] = std::make_shared<XrtData>(
          GetEffectiveDevice(devices[replica]), outputs[i].scalar<int64>()(),
          exec_ops[replica].result_shape,
          [this](XrtData* xrt_data) { ReleaseXrtData(xrt_data); });
    }
  }
  return results;
}

std::vector<std::shared_ptr<ComputationClient::Data>>
XrtComputationClient::ExecuteParallel(
    tensorflow::gtl::ArraySlice<const XlaComputation> computations,
    const std::vector<std::vector<Data*>>& arguments,
    tensorflow::gtl::ArraySlice<const string> devices,
    tensorflow::gtl::ArraySlice<const Shape* const> output_shapes) {
  metrics::TimedSection timed(ExecuteParallelMetric());

  XrtSessionCache::SessionMap session_map;
  tensorflow::ClientSession::FeedType feed_inputs;
  std::vector<ExecuteContext> exec_ops =
      CreateExecuteOps(&session_map, computations, arguments, output_shapes,
                       devices, &feed_inputs);
  std::vector<const XlaComputation*> computations_pointers;
  for (auto& computation : computations) {
    computations_pointers.push_back(&computation);
  }
  return RunComputations(session_map, exec_ops, computations_pointers, devices,
                         feed_inputs);
}

std::vector<std::vector<std::shared_ptr<ComputationClient::Data>>>
XrtComputationClient::DeconstructTuple(
    tensorflow::gtl::ArraySlice<const std::shared_ptr<Data>> tuples) {
  metrics::TimedSection timed(DeconstructTupleMetric());

  XrtSessionCache::SessionMap session_map;
  std::map<XrtSession*, SessionWork> session_work_map;
  std::vector<int64> tuple_elements_count(tuples.size());
  tensorflow::ClientSession::FeedType feed_inputs;
  for (size_t i = 0; i < tuples.size(); ++i) {
    const XrtData& xrt_data = dynamic_cast<const XrtData&>(*tuples[i]);
    XrtSession* session = GetSessionForDevice(xrt_data.device(), &session_map);
    SessionWork* session_work = &session_work_map[session];
    session_work->index_mapping.push_back(i);

    tensorflow::Scope device_scope =
        session->root()->WithDevice(TorchDeviceToXrtDevice(xrt_data.device()));
    int64 count = ShapeUtil::TupleElementCount(xrt_data.shape());
    tuple_elements_count[i] = count;
    for (int64 j = 0; j < count; ++j) {
      const XrtSession::CachedNode& cached_node =
          GetSubTupleNode(session, device_scope, xrt_data.device());
      feed_inputs.insert({cached_node.holders[0], xrt_data.handle});
      tensorflow::Tensor index_tensor(tensorflow::DT_INT32,
                                      tensorflow::TensorShape({1}));
      index_tensor.flat<tensorflow::int32>()(0) = j;
      feed_inputs.insert({cached_node.holders[1], index_tensor});
      session_work->outputs_handles.push_back(*cached_node.output);
    }
  }

  std::vector<std::vector<std::shared_ptr<Data>>> results(tuples.size());
  for (auto& session_work : session_work_map) {
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session_work.first->session()->Run(
        feed_inputs, session_work.second.outputs_handles, &outputs));
    XLA_CHECK_EQ(outputs.size(), session_work.second.outputs_handles.size());

    size_t output_index = 0;
    for (auto li : session_work.second.index_mapping) {
      const XrtData& xrt_data = dynamic_cast<const XrtData&>(*tuples[li]);
      std::vector<std::shared_ptr<Data>> tuple_results;
      for (size_t i = 0; i < tuple_elements_count[li]; ++i, ++output_index) {
        tuple_results.push_back(std::make_shared<XrtData>(
            xrt_data.device(), outputs[output_index].scalar<int64>()(),
            ShapeUtil::GetTupleElementShape(xrt_data.shape(), i),
            [this](XrtData* xrt_data) { ReleaseXrtData(xrt_data); }));
      }
      results[li] = std::move(tuple_results);
    }
  }
  return results;
}

XrtSession* XrtComputationClient::GetSessionForTarget(
    const string& target, XrtSessionCache::SessionMap* session_map) {
  return session_cache_.GetSession(target, session_map);
}

XrtSession* XrtComputationClient::GetSessionForXrtDevice(
    const string& xrt_device, XrtSessionCache::SessionMap* session_map) {
  auto worker_hostport = GetWorkerForXrtDevice(xrt_device);
  return GetSessionForTarget(worker_hostport.second, session_map);
}

XrtSession* XrtComputationClient::GetSessionForDevice(
    const string& device, XrtSessionCache::SessionMap* session_map) {
  return GetSessionForXrtDevice(TorchDeviceToXrtDevice(device), session_map);
}

string XrtComputationClient::GetEffectiveDevice(const string& device) const {
  if (device.empty()) {
    return options_.default_device;
  }
  if (device[0] == ':') {
    // Allow devices with ordinal only specification, to expand from the default
    // device type.
    auto pos = options_.default_device.find(':');
    XLA_CHECK_NE(pos, string::npos) << options_.default_device;
    return options_.default_device.substr(0, pos) + device;
  }
  return device;
}

const string& XrtComputationClient::TorchDeviceToXrtDevice(
    const string& device) const {
  auto device_target = options_.device_map.find(GetEffectiveDevice(device));
  XLA_CHECK(device_target != options_.device_map.end())
      << "Unable to find device: " << device;
  return device_target->second;
}

std::unique_ptr<xrt::XLAComputation> XrtComputationClient::CreateXrtComputation(
    const XlaComputation& computation, int64 num_replicas,
    tensorflow::gtl::ArraySlice<const string> devices,
    const Shape* output_shape) const {
  XLA_CHECK_EQ(num_replicas, devices.size());
  std::unique_ptr<xrt::XLAComputation> xrt_computation(
      new xrt::XLAComputation());
  auto config = xrt_computation->mutable_config();
  config->set_num_replicas(num_replicas);
  config->set_num_cores_per_replica(1);
  if (num_replicas > 1) {
    auto device_assignment = config->mutable_device_assignment();
    auto computation_device = device_assignment->add_computation_devices();
    for (int64 i = 0; i < num_replicas; ++i) {
      const string& xrt_device = TorchDeviceToXrtDevice(devices[i]);
      const auto& core_coords = GetDeviceMeshCoords(xrt_device);
      auto replica_device = computation_device->add_replica_devices();
      for (auto coord : core_coords) {
        replica_device->add_value(coord);
      }
    }
  }
  *config->mutable_program_shape() =
      computation.GetProgramShape().ValueOrDie().ToProto();
  if (output_shape != nullptr) {
    *config->mutable_program_shape()->mutable_result() = *output_shape;
  }
  *xrt_computation->mutable_hlo_snapshot() =
      *computation.Snapshot().ValueOrDie();
  return xrt_computation;
}

absl::optional<string> XrtComputationClient::GetArgumentsDevice(
    tensorflow::gtl::ArraySlice<Data*> arguments) const {
  xla_util::Unique<string> unique_device;
  for (size_t i = 0; i < arguments.size(); ++i) {
    XrtData* xrt_data = dynamic_cast<XrtData*>(arguments[i]);
    unique_device.set(xrt_data->device());
  }
  if (!unique_device) {
    return absl::nullopt;
  }
  return *unique_device;
}

void XrtComputationClient::VerifyReplicasDevices(
    const std::vector<std::vector<Data*>>& arguments,
    tensorflow::gtl::ArraySlice<const string> devices) const {
  std::set<string> unique_devices;
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto opt_device = GetArgumentsDevice(arguments[i]);
    if (opt_device) {
      XLA_CHECK_EQ(*opt_device, devices[i]);
    } else {
      opt_device = devices[i];
    }
    XLA_CHECK(unique_devices.insert(*opt_device).second)
        << "Cannot have two different replicas using the same device: "
        << *opt_device;
  }
}

tensorflow::Tensor XrtComputationClient::GetArgumentsInputs(
    tensorflow::gtl::ArraySlice<Data*> arguments, const string& device,
    tensorflow::ClientSession::FeedType* feed_inputs) {
  tensorflow::Tensor inputs_tensor(tensorflow::DT_INT64,
                                   tensorflow::TensorShape({arguments.size()}));
  for (size_t i = 0; i < arguments.size(); ++i) {
    XrtData* xrt_data = dynamic_cast<XrtData*>(arguments[i]);
    XLA_CHECK_EQ(device, xrt_data->device());
    inputs_tensor.flat<tensorflow::int64>()(i) = xrt_data->handle;
  }
  return inputs_tensor;
}

std::vector<XrtComputationClient::ExecuteContext>
XrtComputationClient::CreateExecuteOps(
    XrtSessionCache::SessionMap* session_map,
    tensorflow::gtl::ArraySlice<const XlaComputation> computations,
    const std::vector<std::vector<Data*>>& arguments,
    tensorflow::gtl::ArraySlice<const Shape* const> output_shapes,
    tensorflow::gtl::ArraySlice<const string> devices,
    tensorflow::ClientSession::FeedType* feed_inputs) {
  std::vector<ExecuteContext> exec_ops;
  VerifyReplicasDevices(arguments, devices);
  for (size_t i = 0; i < computations.size(); ++i) {
    const XlaComputation& computation = computations[i];
    const Shape* output_shape = output_shapes[i];
    ProgramShape program_shape;
    if (output_shape == nullptr) {
      program_shape = computation.GetProgramShape().ValueOrDie();
      output_shape = &program_shape.result();
    }
    auto xrt_computation = CreateXrtComputation(computation, /*num_replicas=*/1,
                                                {devices[i]}, output_shape);

    auto inputs = GetArgumentsInputs(arguments[i], devices[i], feed_inputs);
    const string& xrt_device = TorchDeviceToXrtDevice(devices[i]);
    XrtSession* session = GetSessionForXrtDevice(xrt_device, session_map);
    tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
    const XrtSession::CachedNode& cached_node =
        GetCompileExecuteNode(session, device_scope, devices[i]);
    feed_inputs->insert(
        {cached_node.holders[0], xrt_computation->SerializeAsString()});

    xrt::XRTExecutionConfig exec_config;
    exec_config.set_core_index_in_replica(0);
    exec_config.set_release_input_handles(false);
    exec_config.set_release_compilation_handle(true);
    feed_inputs->insert(
        {cached_node.holders[1], exec_config.SerializeAsString()});
    feed_inputs->insert({cached_node.holders[2], inputs});

    exec_ops.emplace_back(*cached_node.output, *output_shape);
  }
  return exec_ops;
}

std::vector<XrtComputationClient::ExecuteContext>
XrtComputationClient::CreateExecuteOps(
    XrtSessionCache::SessionMap* session_map, const XlaComputation& computation,
    const std::vector<std::vector<Data*>>& arguments, const Shape* output_shape,
    tensorflow::gtl::ArraySlice<const string> devices,
    tensorflow::ClientSession::FeedType* feed_inputs) {
  ProgramShape program_shape;
  if (output_shape == nullptr) {
    program_shape = computation.GetProgramShape().ValueOrDie();
    output_shape = &program_shape.result();
  }
  VerifyReplicasDevices(arguments, devices);
  auto xrt_computation = CreateXrtComputation(computation, arguments.size(),
                                              devices, output_shape);

  absl::optional<tensorflow::ops::Placeholder> computation_holder;
  std::vector<ExecuteContext> exec_ops;
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto inputs = GetArgumentsInputs(arguments[i], devices[i], feed_inputs);
    const string& xrt_device = TorchDeviceToXrtDevice(devices[i]);
    XrtSession* session = GetSessionForXrtDevice(xrt_device, session_map);
    tensorflow::Scope device_scope = session->root()->WithDevice(xrt_device);
    const XrtSession::CachedNode& cached_node =
        GetCompileExecuteNode(session, device_scope, devices[i]);
    feed_inputs->insert(
        {cached_node.holders[0], xrt_computation->SerializeAsString()});

    xrt::XRTExecutionConfig exec_config;
    exec_config.set_core_index_in_replica(0);
    exec_config.set_release_input_handles(false);
    exec_config.set_release_compilation_handle(true);
    feed_inputs->insert(
        {cached_node.holders[1], exec_config.SerializeAsString()});
    feed_inputs->insert({cached_node.holders[2], inputs});

    exec_ops.emplace_back(*cached_node.output, *output_shape);
  }
  return exec_ops;
}

void XrtComputationClient::ReleaseHandles(
    tensorflow::gtl::ArraySlice<const DeviceHandle> handles) {
  metrics::TimedSection timed(ReleaseHandlesTimeMetric());

  struct SessionReleases {
    tensorflow::ClientSession::FeedType feed_inputs;
    std::vector<tensorflow::Operation> releases;
  };
  XrtSessionCache::SessionMap session_map;
  std::map<XrtSession*, SessionReleases> session_releases_map;
  for (auto& handle : handles) {
    XrtSession* session = GetSessionForDevice(handle.device, &session_map);
    SessionReleases* release = &session_releases_map[session];
    tensorflow::Scope device_scope =
        session->root()->WithDevice(TorchDeviceToXrtDevice(handle.device));
    const XrtSession::CachedNode& cached_node =
        GetReleaseAllocationHandleNode(session, device_scope, handle.device);
    release->feed_inputs.insert({cached_node.holders[0], handle.handle});
    release->releases.push_back(*cached_node.operation);
  }
  for (const auto& session_releases : session_releases_map) {
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session_releases.first->session()->Run(
        session_releases.second.feed_inputs, {},
        session_releases.second.releases, &outputs));
  }
  ReleaseHandlesMetric()->AddSample(handles.size());
}

void XrtComputationClient::StartHandleReleaser() {
  triggered_task_.reset(
      new xla_util::TriggeredTask([this]() { HandleReleaser(); }));
}

void XrtComputationClient::HandleReleaser() {
  std::vector<DeviceHandle> released_handles;
  {
    std::lock_guard<std::mutex> lock(lock_);
    released_handles.swap(released_handles_);
  }
  if (!released_handles.empty()) {
    ReleaseHandles(released_handles);
  }
}

void XrtComputationClient::ReleaseXrtData(XrtData* xrt_data) {
  {
    std::lock_guard<std::mutex> lock(lock_);
    xrt_data->Release();
    released_handles_.emplace_back(xrt_data->device(), xrt_data->handle);
  }
  triggered_task_->Activate();
}

std::pair<XrtComputationClient::Worker, string>
XrtComputationClient::GetWorkerForXrtDevice(const string& xrt_device) const {
  tensorflow::DeviceNameUtils::ParsedName parsed_device;
  XLA_CHECK(
      tensorflow::DeviceNameUtils::ParseFullName(xrt_device, &parsed_device) &&
      parsed_device.has_job && parsed_device.has_task)
      << xrt_device;

  auto worker_hostport =
      options_.workers_map.find(Worker(parsed_device.job, parsed_device.task));
  XLA_CHECK(worker_hostport != options_.workers_map.end()) << xrt_device;
  return std::pair<Worker, string>(worker_hostport->first,
                                   worker_hostport->second);
}

std::pair<XrtComputationClient::Worker, string>
XrtComputationClient::GetWorkerForDevice(const string& device) const {
  return GetWorkerForXrtDevice(TorchDeviceToXrtDevice(device));
}

const std::vector<int>& XrtComputationClient::GetDeviceMeshCoords(
    const string& xrt_device) const {
  auto it = device_mesh_coords_.find(xrt_device);
  if (it == device_mesh_coords_.end()) {
    LOG(FATAL) << "Missing mesh coordinates for device: " << xrt_device;
  }
  return it->second;
}

tensorflow::tpu::TopologyProto XrtComputationClient::InitializeAndFetchTopology(
    const string& xrt_device) {
  auto worker_hostport = GetWorkerForXrtDevice(xrt_device);
  LOG(INFO) << "Initializing TPU system for worker "
            << worker_hostport.first.name << ":"
            << worker_hostport.first.task_no << " at "
            << worker_hostport.second;
  string system_device =
      absl::StrCat("/job:", worker_hostport.first.name,
                   "/replica:0/task:", worker_hostport.first.task_no,
                   "/device:TPU_SYSTEM:0");
  XrtSessionCache::SessionMap session_map;
  XrtSession* session =
      GetSessionForTarget(worker_hostport.second, &session_map);
  tensorflow::Scope tpu_system_scope =
      session->root()->WithDevice(system_device);
  const auto unique_name =
      tpu_system_scope.GetUniqueNameForOp("ConfigureDistributedTPU");
  auto builder = tensorflow::NodeBuilder(unique_name, "ConfigureDistributedTPU")
                     .Attr("embedding_config", "")
                     .Attr("tpu_embedding_config", "")
                     .Attr("is_global_init", false);
  tpu_system_scope.UpdateBuilder(&builder);

  tensorflow::Node* result;
  session->root()->UpdateStatus(
      builder.Finalize(tpu_system_scope.graph(), &result));
  TF_CHECK_OK(tpu_system_scope.status());
  session->root()->UpdateStatus(tpu_system_scope.DoShapeInference(result));

  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session->root()->status());
  TF_CHECK_OK(
      session->session()->Run({tensorflow::Output(result, 0)}, &outputs));
  XLA_CHECK_EQ(outputs.size(), 1);

  tensorflow::tpu::TopologyProto topology_proto;
  XLA_CHECK(topology_proto.ParseFromString(outputs[0].scalar<string>()()));
  return topology_proto;
}

void XrtComputationClient::InitializeDevices() {
  auto it = options_.device_map.find("TPU:0");
  if (it != options_.device_map.end()) {
    tensorflow::tpu::TopologyProto topology_proto =
        InitializeAndFetchTopology(it->second);
    LOG(INFO) << "TPU topology: " << topology_proto.DebugString();

    tensorflow::DeviceNameUtils::ParsedName parsed_device;
    XLA_CHECK(tensorflow::DeviceNameUtils::ParseFullName(it->second,
                                                         &parsed_device) &&
              parsed_device.has_job)
        << it->second;
    string tpu_job_name = parsed_device.job;
    for (const auto& dev_target : options_.device_map) {
      XLA_CHECK(tensorflow::DeviceNameUtils::ParseFullName(dev_target.second,
                                                           &parsed_device) &&
                parsed_device.has_job && parsed_device.has_task &&
                parsed_device.has_id)
          << dev_target.second;
      if (parsed_device.job != tpu_job_name) {
        continue;
      }
      XLA_CHECK_LE(parsed_device.task, topology_proto.num_tasks());
      XLA_CHECK_LE(parsed_device.id, topology_proto.num_tpu_devices_per_task());
      // The topology proto 'device_coordinates' is a linear list of
      // [num_tasks][devices_per_task][mesh_shape_size] coordinates, where the
      // mesh coordinates are usually [x, y, c] ('x' and 'y' being the spatial
      // chip coordinated and 'c' the core number).
      int64 base_index = parsed_device.task *
                             topology_proto.num_tpu_devices_per_task() *
                             topology_proto.mesh_shape_size() +
                         parsed_device.id * topology_proto.mesh_shape_size();
      std::vector<int> device_mesh_coords(topology_proto.mesh_shape_size());
      for (int i = 0; i < topology_proto.mesh_shape_size(); ++i) {
        device_mesh_coords[i] =
            topology_proto.device_coordinates(base_index + i);
      }
      device_mesh_coords_.insert(
          {dev_target.second, std::move(device_mesh_coords)});
    }
  }
}

string XrtComputationClient::GetDefaultDevice() const {
  return options_.default_device;
}

const XrtSession::CachedNode& XrtComputationClient::GetCompileExecuteNode(
    XrtSession* session, const tensorflow::Scope& scope, const string& device) {
  static const string op_name("XrtCompileExecute");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING),
         tensorflow::ops::Placeholder(
             scope, tensorflow::DT_INT64,
             tensorflow::ops::Placeholder::Shape({-1}))});
    auto computation_handle = tensorflow::ops::XRTCompile(scope, holders[0]);
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTExecute(scope, computation_handle.handle,
                                    holders[1],
                                    {tensorflow::Output(holders[2])}),
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetReadNode(
    XrtSession* session, const tensorflow::Scope& scope, const string& device) {
  static const string op_name("XrtRead");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTReadLiteral(scope, holders[0]),
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetAllocateNode(
    XrtSession* session, const tensorflow::Scope& scope, const string& device) {
  static const string op_name("XrtAllocate");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_STRING)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTAllocate(scope, holders[0]), std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode&
XrtComputationClient::GetReleaseAllocationHandleNode(
    XrtSession* session, const tensorflow::Scope& scope, const string& device) {
  static const string op_name("XrtReleaseAllocationHandle");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64)});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTReleaseAllocationHandle(scope, holders[0]),
        std::move(holders)));
  }
  return cache->Get();
}

const XrtSession::CachedNode& XrtComputationClient::GetSubTupleNode(
    XrtSession* session, const tensorflow::Scope& scope, const string& device) {
  static const string op_name("XrtSubTuple");
  XrtSession::NodeCache* cache =
      session->GetNodeCache(XrtSession::GetCacheKey(op_name, device));
  if (cache->Empty()) {
    std::vector<tensorflow::ops::Placeholder> holders(
        {tensorflow::ops::Placeholder(scope, tensorflow::DT_INT64),
         tensorflow::ops::Placeholder(
             scope, tensorflow::DT_INT32,
             tensorflow::ops::Placeholder::Shape({1}))});
    cache->Add(std::make_shared<XrtSession::CachedNode>(
        tensorflow::ops::XRTSubTuple(scope, holders[0], holders[1]),
        std::move(holders)));
  }
  return cache->Get();
}

std::vector<std::vector<ComputationClient::Data*>>
XrtComputationClient::BuildParallelArguments(
    tensorflow::gtl::ArraySlice<Data*> arguments) {
  std::vector<std::vector<ComputationClient::Data*>> para_arguments(1);
  para_arguments[0].insert(para_arguments[0].end(), arguments.begin(),
                           arguments.end());
  return para_arguments;
}

void XrtComputationClient::MaybeCreateLocalService(
    const XrtComputationClient::Options& options) {
  static const string grpc_root("grpc://localhost:");
  int task_index = -1;
  string job_name;
  string cluster_spec;
  for (auto& worker_target : options.workers_map) {
    if (worker_target.second.compare(0, grpc_root.size(), grpc_root) == 0 &&
        worker_target.first.name == "localservice") {
      job_name = worker_target.first.name;
      task_index = worker_target.first.task_no;
      cluster_spec = absl::StrCat(
          worker_target.first.name,
          "|localhost:", worker_target.second.substr(grpc_root.size()));
    }
  }
  if (!cluster_spec.empty()) {
    XrtLocalService* service =
        new XrtLocalService(cluster_spec, job_name, task_index);
    service->Start();
  }
}

}  // namespace xla
