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
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace triton { namespace perfanalyzer {

// Perf Analyzer command line parameters.
// PAParams are used to initialize PerfAnalyzer and track configuration
//
struct PerfAnalyzerParameters {
  bool verbose = false;
  bool extra_verbose = false;
  bool streaming = false;
  size_t max_threads = 4;
  bool max_threads_specified = false;
  size_t sequence_length = 20; // average length of a sentence
  int32_t percentile = -1;
  std::vector<std::string> user_data;
};

using PAParamsPtr = std::shared_ptr<PerfAnalyzerParameters>; 

class CLParser {
 public:
  CLParser() : params_(new PerfAnalyzerParameters{}){}

  // Parse command line arguements into a parameters struct
  //
  PAParamsPtr parse(int argc, char** argv);

 private:
  PAParamsPtr params_;

  std::string format_message(std::string str, int offset) const;
  void usage(char** argv, const std::string& msg = std::string());
  void parse_command_line(int argc, char** argv);
  void initialize_options();
  void verify_options();
};
}} // namespace triton::perfanalyzer
