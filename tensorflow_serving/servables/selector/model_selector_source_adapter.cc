#include "tensorflow_serving/servables/selector/model_selector_source_adapter.h"

#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace serving {

Status ParseProtoTextFile(const string& file,
                                     google::protobuf::Message* message) {
    std::unique_ptr<tensorflow::ReadOnlyMemoryRegion> file_data;
    TF_RETURN_IF_ERROR(
            tensorflow::Env::Default()->NewReadOnlyMemoryRegionFromFile(file,
                                                                        &file_data));
    string file_data_str(static_cast<const char*>(file_data->data()),
                         file_data->length());
    if (tensorflow::protobuf::TextFormat::ParseFromString(file_data_str,
                                                          message)) {
        return tensorflow::Status::OK();
    } else {
        return tensorflow::errors::InvalidArgument("Invalid protobuf file: '", file,
                                                   "'");
    }
}

template <typename ProtoType>
ProtoType ReadProtoFromFile(const string& file) {
    ProtoType proto;
    TF_CHECK_OK(ParseProtoTextFile(file, &proto));
    return proto;
}

Status LoadModelSelectorFromFile(const string& path,
                                 const string& file_name,
                                 std::unique_ptr<ModelSelector>* selector) {
    auto config = ReadProtoFromFile<ModelSelectorConfig>(path + "/" + file_name);
    std::vector<Candidate> candidates;
    candidates.reserve(config.candidates_size());
    float sum_weights = 0;
    for (auto& candidate : config.candidates()) {
        sum_weights += candidate.weight();
        candidates.push_back(candidate);
    }
    selector->reset(new ModelSelector(sum_weights, candidates, config.signaturegroups()));
    return Status::OK();
}

ModelSelectorSourceAdapter::ModelSelectorSourceAdapter(
    const ModelSelectorSourceAdapterConfig& config)
    : SimpleLoaderSourceAdapter<StoragePath, ModelSelector>(
        [config](const StoragePath& path, std::unique_ptr<ModelSelector>* selector) {
            return LoadModelSelectorFromFile(path, config.filename(), selector);
        },
        SimpleLoaderSourceAdapter<StoragePath, ModelSelector>::EstimateNoResources()) {}

ModelSelectorSourceAdapter::~ModelSelectorSourceAdapter() { Detach(); }

class ModelSelectorSourceAdapterCreator {
public:
    static Status Create(
            const ModelSelectorSourceAdapterConfig& config,
            std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>* adapter) {
        adapter->reset(new ModelSelectorSourceAdapter(config));
        return Status::OK();
    }
};

REGISTER_STORAGE_PATH_SOURCE_ADAPTER(ModelSelectorSourceAdapterCreator, ModelSelectorSourceAdapterConfig);

} // namespace serving
} // namespace tensorlfow
