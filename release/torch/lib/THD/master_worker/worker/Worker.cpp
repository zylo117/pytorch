#include "../../process_group/General.hpp"
#include "../common/RPC.hpp"
#include "Dispatch.hpp"
#include "Worker.h"
#include "Worker.hpp"

#include <iostream>
#include <stdexcept>

namespace thd {
namespace worker {

std::unique_ptr<WorkerCommandChannel> workerCommandChannel;
std::unordered_map<object_id_type, at::Tensor> workerTensors;
std::unordered_map<object_id_type, std::unique_ptr<at::Storage>> workerStorages;
std::unordered_map<object_id_type, std::unique_ptr<at::Generator>> workerGenerators;

} // namespace worker
} // namespace thd

using namespace thd::rpc;
using namespace thd::worker;

void THDWorkerMain(std::string init_method, int world_size,
                   std::string group_name, int rank) {
  auto config = thd::getInitConfig(init_method, world_size, group_name, rank);
  std::unique_ptr<RPCMessage> command;
  workerCommandChannel.reset(new thd::WorkerCommandChannel(config));
  if (!workerCommandChannel->init()) {
    return;
  }

  while (true) {
    command = workerCommandChannel->recvMessage();
    try {
      execute(std::move(command));
    } catch (std::exception& e) {
      std::cerr << "WORKER ERROR: " << e.what() << std::endl;
      workerCommandChannel->sendError(e.what());
      ::exit(1);
    }
  }
}
