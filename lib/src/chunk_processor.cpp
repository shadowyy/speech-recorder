#include "chunk_processor.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <locale>
#include <memory>

#ifndef MIN
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif

namespace speechrecorder {

static std::mutex ortMutex_;
static std::unique_ptr<Ort::Env> ortEnv_;
static std::unique_ptr<Ort::MemoryInfo> ortMemory_;
static std::unique_ptr<Ort::Session> ortSession_;

ChunkProcessor::ChunkProcessor(std::string modelPath,
                               ChunkProcessorOptions options)
    : options_(options),
      queue_(),
      stopped_(false),
      microphone_(options.device, options.samplesPerFrame, options.sampleRate,
                  &queue_),
      webrtcVad_(options.webrtcVadLevel, options.sampleRate) {
  queueThread_ = std::thread([&, modelPath] {
    setlocale(LC_ALL, "pt_BR.UTF-8");

    ortMutex_.lock();
    if (!ortSession_) {
      ortEnv_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                           "SpeechRecorder::ChunkProcessor");
      ortMemory_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(
          OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault));

      Ort::SessionOptions sessionOptions;
      sessionOptions.SetIntraOpNumThreads(1);
#ifdef _WIN32
      std::filesystem::path unicodeModelPath(modelPath.begin(),
                                             modelPath.end());
      ortSession_ = std::make_unique<Ort::Session>(
          *ortEnv_, unicodeModelPath.c_str(), sessionOptions);
#else
      ortSession_ = std::make_unique<Ort::Session>(*ortEnv_, modelPath.c_str(),
                                                   sessionOptions);
#endif
    }
    ortMutex_.unlock();
    while (true) {
      short *audio;
      queue_.wait_dequeue(audio);
      // null pointer means the destructor wants us to stop the thread.
      if (audio == nullptr) {
        return;
      }
      if (!stopped_) {
        Process(audio);
      }
    }
  });
}

ChunkProcessor::~ChunkProcessor() {
  // shutdown the queue thread.
  stopped_ = true;
  queue_.enqueue(nullptr);
  queueThread_.join();

  if (stopThread_.joinable()) {
    stopThread_.join();
  }
  if (startThread_.joinable()) {
    startThread_.join();
  }
}

int agcProcess(int16_t *buffer, uint32_t sampleRate, size_t samplesCount,
               int16_t agcMode) {
  if (buffer == nullptr) return -1;
  if (samplesCount == 0) return -1;
  WebRtcAgcConfig agcConfig;
  agcConfig.compressionGaindB = 9;  // default 9 dB
  agcConfig.limiterEnable = 1;      // default kAgcTrue (on)
  agcConfig.targetLevelDbfs = 3;    // default 3 (-3 dBOv)
  int minLevel = 0;
  int maxLevel = 255;
  size_t samples = MIN(160, sampleRate / 100);
  if (samples == 0) return -1;
  const int maxSamples = 320;
  int16_t *input = buffer;
  size_t nTotal = (samplesCount / samples);
  void *agcInst = WebRtcAgc_Create();
  if (agcInst == NULL) return -1;
  int status = WebRtcAgc_Init(agcInst, minLevel, maxLevel, agcMode, sampleRate);
  if (status != 0) {
    printf("WebRtcAgc_Init fail\n");
    WebRtcAgc_Free(agcInst);
    return -1;
  }
  status = WebRtcAgc_set_config(agcInst, agcConfig);
  if (status != 0) {
    printf("WebRtcAgc_set_config fail\n");
    WebRtcAgc_Free(agcInst);
    return -1;
  }
  size_t num_bands = 1;
  int inMicLevel, outMicLevel = -1;
  int16_t out_buffer[maxSamples];
  int16_t *out16 = out_buffer;
  uint8_t saturationWarning =
      1;  // 是否有溢出发生，增益放大以后的最大值超过了65536
  int16_t echo = 0;  // 增益放大是否考虑回声影响
  for (int i = 0; i < nTotal; i++) {
    inMicLevel = 0;
    int nAgcRet =
        WebRtcAgc_Process(agcInst, (const int16_t *const *)&input, num_bands,
                          samples, (int16_t *const *)&out16, inMicLevel,
                          &outMicLevel, echo, &saturationWarning);

    if (nAgcRet != 0) {
      printf("failed in WebRtcAgc_Process\n");
      WebRtcAgc_Free(agcInst);
      return -1;
    }
    memcpy(input, out_buffer, samples * sizeof(int16_t));
    input += samples;
  }

  const size_t remainedSamples = samplesCount - nTotal * samples;
  if (remainedSamples > 0) {
    if (nTotal > 0) {
      input = input - samples + remainedSamples;
    }

    inMicLevel = 0;
    int nAgcRet =
        WebRtcAgc_Process(agcInst, (const int16_t *const *)&input, num_bands,
                          samples, (int16_t *const *)&out16, inMicLevel,
                          &outMicLevel, echo, &saturationWarning);

    if (nAgcRet != 0) {
      printf("failed in WebRtcAgc_Process during filtering the last chunk\n");
      WebRtcAgc_Free(agcInst);
      return -1;
    }
    memcpy(&input[samples - remainedSamples],
           &out_buffer[samples - remainedSamples],
           remainedSamples * sizeof(int16_t));
    input += samples;
  }

  WebRtcAgc_Free(agcInst);
  return 1;
}

void ChunkProcessor::Process(short *input) {
  std::vector<short> frame;
  agcProcess(input, options_.sampleRate, options_.samplesPerFrame,
             kAgcModeAdaptiveDigital);

  const short *iterator = (const short *)input;
  unsigned long long sum = 0;
  for (unsigned long i = 0; i < options_.samplesPerFrame; i++) {
    const short value = *iterator++;
    frame.push_back(value);
    leadingBuffer_.push_back(value);
    sileroBuffer_.push_back((float)value / (float)SHRT_MAX);
    webrtcVadBuffer_.push_back(value);
    sum += value * value;
  }

  double volume = sqrt((double)sum / (double)options_.samplesPerFrame);
  if (leadingBuffer_.size() >
      options_.leadingBufferFrames * options_.samplesPerFrame) {
    leadingBuffer_.erase(
        leadingBuffer_.begin(),
        leadingBuffer_.begin() +
            (leadingBuffer_.size() -
             (options_.leadingBufferFrames * options_.samplesPerFrame)));
  }

  if (sileroBuffer_.size() > options_.sileroVadBufferSize) {
    sileroBuffer_.erase(sileroBuffer_.begin(),
                        sileroBuffer_.begin() + (sileroBuffer_.size() -
                                                 options_.sileroVadBufferSize));
  }

  // typically, the number of samples per frame will be larger than the
  // webrtcvad buffer size, so continually append the new audio to the end of
  // the buffer, and process the buffer from left to right until it's too small
  // for a webrtcvad call
  while (webrtcVadBuffer_.size() >= options_.webrtcVadBufferSize) {
    std::vector<short> buffer(
        webrtcVadBuffer_.begin(),
        webrtcVadBuffer_.begin() + options_.webrtcVadBufferSize);
    webrtcVadResults_.push_back(
        webrtcVad_.Process(buffer.data(), options_.webrtcVadBufferSize));
    webrtcVadBuffer_.erase(
        webrtcVadBuffer_.begin(),
        webrtcVadBuffer_.begin() + options_.webrtcVadBufferSize);
  }

  if (webrtcVadResults_.size() > options_.webrtcVadResultsSize) {
    webrtcVadResults_.erase(
        webrtcVadResults_.begin(),
        webrtcVadResults_.begin() +
            (webrtcVadResults_.size() - options_.webrtcVadResultsSize));
  }

  if (framesUntilSileroVad_ > 0) {
    framesUntilSileroVad_--;
  }

  // if we're speaking or any past webrtcvad result within the window is true,
  // then use the result from the silero vad
  double probability = 0.0;
  if (speaking_ || webrtcVadResults_.size() != options_.webrtcVadResultsSize ||
      std::any_of(webrtcVadResults_.begin(), webrtcVadResults_.end(),
                  [](bool e) { return e; })) {
    if (framesUntilSileroVad_ == 0) {
      framesUntilSileroVad_ = options_.sileroVadRateLimit;

      std::vector<int64_t> inputDimensions;
      inputDimensions.push_back(1);
      inputDimensions.push_back(sileroBuffer_.size());

      std::vector<Ort::Value> inputTensors;
      inputTensors.push_back(Ort::Value::CreateTensor<float>(
          *ortMemory_, sileroBuffer_.data(), sileroBuffer_.size(),
          inputDimensions.data(), inputDimensions.size()));

      std::vector<float> outputTensorValues(2);
      std::vector<int64_t> outputDimensions;
      outputDimensions.push_back(1);
      outputDimensions.push_back(2);

      std::vector<Ort::Value> outputTensors;
      outputTensors.push_back(Ort::Value::CreateTensor<float>(
          *ortMemory_, outputTensorValues.data(), outputTensorValues.size(),
          outputDimensions.data(), outputDimensions.size()));

      std::vector<const char *> inputNames{"input"};
      std::vector<const char *> outputNames{"output"};
      ortSession_->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                       inputTensors.data(), 1, outputNames.data(),
                       outputTensors.data(), 1);

      sileroVadProbability_ = outputTensorValues[1];
    }

    probability = sileroVadProbability_;
  }

  bool speaking = speaking_ ? probability > options_.sileroVadSilenceThreshold
                            : probability > options_.sileroVadSpeakingThreshold;
  if (speaking) {
    consecutiveSilence_ = 0;
    consecutiveSpeaking_++;
  } else {
    consecutiveSilence_++;
    consecutiveSpeaking_ = 0;
  }

  if (!speaking_ &&
      consecutiveSpeaking_ == options_.consecutiveFramesForSpeaking) {
    speaking_ = true;
    if (options_.onChunkStart != nullptr) {
      options_.onChunkStart(leadingBuffer_);
    }
  }

  if (options_.onAudio != nullptr) {
    options_.onAudio(frame, speaking_, volume, speaking, probability,
                     consecutiveSilence_);
  }

  if (speaking_ &&
      consecutiveSilence_ == options_.consecutiveFramesForSilence) {
    speaking_ = false;
    leadingBuffer_.clear();
    if (options_.onChunkEnd != nullptr) {
      options_.onChunkEnd();
    }
  }
}

void ChunkProcessor::Reset() {
  consecutiveSilence_ = 0;
  consecutiveSpeaking_ = 0;
  framesUntilSileroVad_ = 0;
  leadingBuffer_.clear();
  speaking_ = false;
  webrtcVad_.Reset();
  webrtcVadBuffer_.clear();
  webrtcVadResults_.clear();
  short *audio;
  while (queue_.try_dequeue(audio)) {
  }
}

void ChunkProcessor::Start() {
  if (stopThread_.joinable()) {
    stopThread_.join();
  }
  toggleLock_.lock();
  startThread_ = std::thread([&] {
    Reset();
    microphone_.Start();
    stopped_ = false;
    toggleLock_.unlock();
  });
}

void ChunkProcessor::Stop() {
  if (startThread_.joinable()) {
    startThread_.join();
  }
  toggleLock_.lock();
  stopThread_ = std::thread([&] {
    stopped_ = true;
    microphone_.Stop();
    toggleLock_.unlock();
  });
}

}  // namespace speechrecorder
