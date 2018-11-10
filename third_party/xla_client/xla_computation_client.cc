#include "tensorflow/compiler/xla/xla_client/xla_computation_client.h"

#include "grpc++/create_channel.h"
#include "grpc++/support/channel_arguments.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/xla_client/unique.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"

namespace xla {

XlaComputationClient::XlaComputationClient(
    XlaComputationClient::Options options)
    : options_(std::move(options)) {
  int64 device_count = 1;
  if (!options_.host_name.empty()) {
    ::grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(-1);
    auto channel = ::grpc::CreateCustomChannel(
        absl::StrCat(options_.host_name, ":", options_.port),
        ::grpc::InsecureChannelCredentials(), ch_args);
    channel->WaitForConnected(gpr_time_add(
        gpr_now(GPR_CLOCK_REALTIME), gpr_time_from_seconds(10, GPR_TIMESPAN)));
    LOG(INFO) << "Channel to '" << options_.host_name
              << "' is connected on port " << options_.port;

    xla_service_ = grpc::XlaService::NewStub(channel);
    stub_.reset(new GRPCStub(xla_service_.get()));
    client_ptr_.reset(new Client(stub_.get()));
    client_ = client_ptr_.get();
    device_count = options_.platform == "TPU" ? 8 : 1;
  } else {
    se::Platform* platform = nullptr;
    if (!options_.platform.empty()) {
      platform = PlatformUtil::GetPlatform(options_.platform).ValueOrDie();
    }
    LOG(INFO) << "Creating XLA computation client for '"
              << (options_.platform.empty() ? "default" : options_.platform)
              << "' platform";
    LocalClient* local_client =
        ClientLibrary::GetOrCreateLocalClient(platform).ValueOrDie();
    device_count = local_client->device_count();
    client_ = local_client;
  }
  device_handles_ = client_->GetDeviceHandles(device_count).ValueOrDie();
}

std::shared_ptr<ComputationClient::Data>
XlaComputationClient::ExecuteComputation(
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<Data*> arguments, const Shape* output_shape) {
  metrics::TimedSection timed(ExecuteMetric());
  FlushReleasedHandles();

  string device;
  std::vector<GlobalData*> arguments_data =
      GetArgumentsData(arguments, &device);
  ExecutionOptions eo;
  *eo.mutable_debug_options() = legacy_flags::GetDebugOptionsFromFlags();
  *eo.add_device_handles() = GetDeviceHandle(device);
  if (output_shape != nullptr) {
    *eo.mutable_shape_with_output_layout() = *output_shape;
  }
  StatusOr<std::unique_ptr<GlobalData>> result_or_status =
      client_->Execute(computation, arguments_data, &eo);
  xrt_util::CheckComputationStatus(result_or_status.status(), {&computation});

  ProgramShape program_shape;
  if (output_shape == nullptr) {
    program_shape = computation.GetProgramShape().ValueOrDie();
    output_shape = &program_shape.result();
  }
  return std::make_shared<XlaData>(
      std::move(result_or_status.ValueOrDie()), device, *output_shape,
      [this](XlaData* xla_data) { ReleaseXlaData(xla_data); });
}

std::unique_ptr<Literal> XlaComputationClient::ExecuteComputationAndTransfer(
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<Data*> arguments, const Shape* output_shape) {
  metrics::TimedSection timed(ExecuteTransferMetric());
  FlushReleasedHandles();

  string device;
  std::vector<GlobalData*> arguments_data =
      GetArgumentsData(arguments, &device);
  ExecutionOptions eo;
  *eo.mutable_debug_options() = legacy_flags::GetDebugOptionsFromFlags();
  *eo.add_device_handles() = GetDeviceHandle(device);
  if (output_shape != nullptr) {
    *eo.mutable_shape_with_output_layout() = *output_shape;
  }
  StatusOr<Literal> result_or_status =
      client_->ExecuteAndTransfer(computation, arguments_data, &eo);
  xrt_util::CheckComputationStatus(result_or_status.status(), {&computation});
  std::unique_ptr<Literal> result(
      new Literal(std::move(result_or_status.ValueOrDie())));
  InboundDataMetric()->AddSample(result->size_bytes());
  return result;
}

std::vector<std::shared_ptr<ComputationClient::Data>>
XlaComputationClient::ExecuteReplicated(
    const XlaComputation& computation,
    const std::vector<std::vector<Data*>>& arguments,
    const Shape* output_shape) {
  metrics::TimedSection timed(ExecuteReplicatedMetric());
  LOG(FATAL) << "ExecuteReplicated() API not yet implemented!";
}

std::vector<std::shared_ptr<ComputationClient::Data>>
XlaComputationClient::ExecuteParallel(
    tensorflow::gtl::ArraySlice<const XlaComputation> computations,
    const std::vector<std::vector<Data*>>& arguments,
    tensorflow::gtl::ArraySlice<const Shape* const> output_shapes) {
  metrics::TimedSection timed(ExecuteParallelMetric());

  std::vector<const XlaComputation*> computations_pointers;
  std::vector<string> devices(computations.size());
  std::vector<Client::XlaComputationInstance> instances;
  for (size_t i = 0; i < computations.size(); ++i) {
    std::vector<GlobalData*> arguments_data =
        GetArgumentsData(arguments[i], &devices[i]);
    ExecutionOptions eo;
    *eo.mutable_debug_options() = legacy_flags::GetDebugOptionsFromFlags();
    *eo.add_device_handles() = GetDeviceHandle(devices[i]);
    if (output_shapes[i] != nullptr) {
      *eo.mutable_shape_with_output_layout() = *output_shapes[i];
    }
    instances.emplace_back(computations[i], std::move(arguments_data),
                           std::move(eo), nullptr);
    computations_pointers.push_back(&computations[i]);
  }

  StatusOr<std::vector<std::unique_ptr<GlobalData>>> results_or_status =
      client_->ExecuteParallel(instances);
  xrt_util::CheckComputationStatus(results_or_status.status(),
                                   computations_pointers);
  std::vector<std::unique_ptr<GlobalData>> exec_results(
      std::move(results_or_status.ValueOrDie()));
  CHECK_EQ(exec_results.size(), computations.size());
  std::vector<std::shared_ptr<Data>> results;
  for (size_t i = 0; i < computations.size(); ++i) {
    ProgramShape program_shape;
    const Shape* output_shape = output_shapes[i];
    if (output_shape == nullptr) {
      program_shape = computations[i].GetProgramShape().ValueOrDie();
      output_shape = &program_shape.result();
    }
    results.push_back(std::make_shared<XlaData>(
        std::move(exec_results[i]), devices[i], *output_shape,
        [this](XlaData* xla_data) { ReleaseXlaData(xla_data); }));
  }
  return results;
}

std::shared_ptr<ComputationClient::Data>
XlaComputationClient::TransferParameterToServer(const Literal& literal,
                                                const string& device) {
  metrics::TimedSection timed(TransferMetric());
  OutboundDataMetric()->AddSample(literal.size_bytes());
  FlushReleasedHandles();

  std::unique_ptr<GlobalData> handle =
      client_->TransferToServer(literal).ValueOrDie();
  return std::make_shared<XlaData>(
      std::move(handle), GetEffectiveDevice(device), literal.shape(),
      [this](XlaData* xla_data) { ReleaseXlaData(xla_data); });
}

StatusOr<std::vector<std::shared_ptr<ComputationClient::Data>>>
XlaComputationClient::DeconstructTuple(const Data& data) {
  metrics::TimedSection timed(DeconstructTupleMetric());
  const XlaData& xla_data = dynamic_cast<const XlaData&>(data);
  TF_ASSIGN_OR_RETURN(auto exploded_tuple,
                      client_->DeconstructTuple(*xla_data.handle));
  std::vector<std::shared_ptr<Data>> tuple;
  for (int64 i = 0; i < exploded_tuple.size(); ++i) {
    tuple.push_back(std::make_shared<XlaData>(
        std::move(exploded_tuple[i]), xla_data.device(),
        ShapeUtil::GetTupleElementShape(xla_data.shape(), i),
        [this](XlaData* xla_data) { ReleaseXlaData(xla_data); }));
  }
  return std::move(tuple);
}

std::vector<GlobalData*> XlaComputationClient::GetArgumentsData(
    tensorflow::gtl::ArraySlice<Data*> arguments, string* device) const {
  xla_util::Unique<string> unique_device;
  std::vector<GlobalData*> arguments_data;
  for (auto data : arguments) {
    XlaData* xla_data = dynamic_cast<XlaData*>(data);
    unique_device.set(xla_data->device());
    arguments_data.push_back(xla_data->handle.get());
  }
  if (device != nullptr) {
    if (unique_device) {
      *device = *unique_device;
    } else {
      *device = GetDefaultDevice();
    }
  }
  return arguments_data;
}

const DeviceHandle& XlaComputationClient::GetDeviceHandle(
    const string& device) const {
  int64 ordinal = GetDeviceOrdinal(device);
  CHECK_LT(ordinal, device_handles_.size()) << device;
  return device_handles_[ordinal];
}

string XlaComputationClient::GetEffectiveDevice(const string& device) const {
  return device.empty() ? GetDefaultDevice() : device;
}

void XlaComputationClient::FlushReleasedHandles() {
  std::vector<std::unique_ptr<GlobalData>> released_handles;
  released_handles.swap(released_handles_);
  GlobalData::Release(std::move(released_handles));
}

void XlaComputationClient::ReleaseXlaData(XlaData* xla_data) {
  released_handles_.push_back(xla_data->Release());
}

string XlaComputationClient::GetDefaultDevice() const {
  return options_.platform + ":0";
}

}  // namespace xla