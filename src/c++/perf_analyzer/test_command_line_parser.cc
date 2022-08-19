// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#include <array>

#include "command_line_parser.h"
#include "doctest.h"

namespace triton { namespace perfanalyzer {

inline void
CHECK_STRING(const char* name, std::string str, const char* val)
{
  CHECK_MESSAGE(
      !str.compare(val), name, " expecting '", val, "', found '", str, "'");
}

TEST_CASE("Testing PerfAnalyzerParameters")
{
  PAParamsPtr params(new PerfAnalyzerParameters{});

  CHECK(params->verbose == false);
  CHECK(params->streaming == false);
  CHECK(params->extra_verbose == false);
  CHECK(params->max_threads == 4);
  CHECK(params->max_threads_specified == false);
  CHECK(params->sequence_length == 20);
  CHECK(params->percentile == -1);
  CHECK(params->user_data.size() == 0);

  CHECK(params->user_data.size() == 0);
  CHECK(params->input_shapes.size() == 0);
  CHECK(params->measurement_window_ms == 5000);
  CHECK(params->using_concurrency_range == false);
  CHECK(params->concurrency_range[0] == 1);
  CHECK(params->concurrency_range[1] == 1);
  CHECK(params->concurrency_range[2] == 1);
  CHECK(params->latency_threshold_ms == NO_LIMIT);
  CHECK(params->stability_threshold == doctest::Approx(0.1));
  CHECK(params->max_trials == 10);
  CHECK(params->zero_input == false);
  CHECK(params->string_length == 128);
  CHECK_STRING("string_data", params->string_data, "");
  CHECK(params->async == false);
  CHECK(params->forced_sync == false);
  CHECK(params->using_request_rate_range == false);
  CHECK(params->request_rate_range[0] == doctest::Approx(1.0));
  CHECK(params->request_rate_range[1] == doctest::Approx(1.0));
  CHECK(params->request_rate_range[2] == doctest::Approx(1.0));
  CHECK(params->num_of_sequences == 4);
  CHECK(params->search_mode == SearchMode::LINEAR);
  CHECK(params->request_distribution == Distribution::CONSTANT);
  CHECK(params->using_custom_intervals == false);
  CHECK_STRING("request_intervals_file", params->request_intervals_file, "");
  CHECK(params->shared_memory_type == NO_SHARED_MEMORY);
  CHECK(params->output_shm_size == 102400);
  CHECK(params->kind == clientbackend::BackendKind::TRITON);
  CHECK_STRING(
      "model_signature_name", params->model_signature_name, "serving_default");
  CHECK(params->using_grpc_compression == false);
  CHECK(
      params->compression_algorithm ==
      clientbackend::GrpcCompressionAlgorithm::COMPRESS_NONE);
  CHECK(params->measurement_mode == MeasurementMode::TIME_WINDOWS);
  CHECK(params->measurement_request_count == 50);
  CHECK_STRING("triton_server_path", params->triton_server_path, "");
  CHECK_STRING("model_repository_path", params->model_repository_path, "");
  CHECK(params->start_sequence_id == 1);
  CHECK(params->sequence_id_range == UINT32_MAX);

  // Come back to this!
  // CHECK(params->ssl_options;  // gRPC and HTTP SSL options

  CHECK(params->verbose_csv == false);
  CHECK(params->enable_mpi == false);
  CHECK(params->trace_options.size() == 0);
  CHECK(params->using_old_options == false);
  CHECK(params->dynamic_concurrency_mode == false);
  CHECK(params->url_specified == false);
  CHECK_STRING("url", params->url, "localhost:8000");
  CHECK_STRING("model_name", params->model_name, "");
  CHECK_STRING("model_version", params->model_version, "");
  CHECK(params->batch_size == 1);
  CHECK(params->using_batch_size == false);
  CHECK(params->concurrent_request_count == 1);
  CHECK(params->protocol == clientbackend::ProtocolType::HTTP);
  CHECK(params->http_headers->size() == 0);
  CHECK(params->max_concurrency == 0);
  CHECK_STRING("filename", params->filename, "");
  CHECK(params->mpi_driver == nullptr);
  CHECK_STRING("memory_type", params->memory_type, "system");
}

TEST_CASE("Testing Command Line Parser")
{
  CLParser parser;

  SUBCASE("with min parameters")
  {
    char* argv[3] = {"test_perf_analyzer", "-m", "my_model"};
    int argc = 3;
    PAParamsPtr p = parser.parse(argc, argv);

    CHECK(p->verbose == false);
    CHECK(p->streaming == false);
    CHECK(p->extra_verbose == false);
    CHECK_STRING("model_name", p->model_name, "my_model");
  }
}

}}  // namespace triton::perfanalyzer
