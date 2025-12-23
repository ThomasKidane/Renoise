/**
 * Native WebRTC AEC3 Node.js Addon
 * Provides real-time echo cancellation using WebRTC's AEC3 algorithm
 */

#include <napi.h>
#include "api/echo_canceller3_factory.h"
#include "api/echo_canceller3_config.h"
#include "audio_processing/include/audio_processing.h"
#include "audio_processing/audio_buffer.h"
#include "audio_processing/high_pass_filter.h"
#include <memory>
#include <vector>

using namespace webrtc;

class AEC3Wrapper : public Napi::ObjectWrap<AEC3Wrapper> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  AEC3Wrapper(const Napi::CallbackInfo& info);
  ~AEC3Wrapper();

private:
  static Napi::FunctionReference constructor;
  
  Napi::Value Process(const Napi::CallbackInfo& info);
  Napi::Value ProcessCapture(const Napi::CallbackInfo& info);
  Napi::Value AnalyzeRender(const Napi::CallbackInfo& info);
  Napi::Value Reset(const Napi::CallbackInfo& info);
  
  std::unique_ptr<EchoControl> echo_controller_;
  std::unique_ptr<HighPassFilter> hp_filter_;
  std::unique_ptr<AudioBuffer> capture_buffer_;
  std::unique_ptr<AudioBuffer> render_buffer_;
  
  int sample_rate_;
  int frame_size_;
  int num_channels_;
  int mode_;
  bool use_hp_filter_;
};

Napi::FunctionReference AEC3Wrapper::constructor;

Napi::Object AEC3Wrapper::Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);
  
  Napi::Function func = DefineClass(env, "AEC3", {
    InstanceMethod("process", &AEC3Wrapper::Process),
    InstanceMethod("processCapture", &AEC3Wrapper::ProcessCapture),
    InstanceMethod("analyzeRender", &AEC3Wrapper::AnalyzeRender),
    InstanceMethod("reset", &AEC3Wrapper::Reset),
  });
  
  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  
  exports.Set("AEC3", func);
  return exports;
}

AEC3Wrapper::AEC3Wrapper(const Napi::CallbackInfo& info) 
    : Napi::ObjectWrap<AEC3Wrapper>(info) {
  Napi::Env env = info.Env();
  
  // Default parameters
  sample_rate_ = 48000;
  frame_size_ = 480;  // 10ms at 48kHz
  num_channels_ = 1;
  mode_ = 7;  // Ultra aggressive by default
  
  // Parse options if provided
  if (info.Length() > 0 && info[0].IsObject()) {
    Napi::Object options = info[0].As<Napi::Object>();
    
    if (options.Has("sampleRate")) {
      sample_rate_ = options.Get("sampleRate").As<Napi::Number>().Int32Value();
    }
    if (options.Has("frameSize")) {
      frame_size_ = options.Get("frameSize").As<Napi::Number>().Int32Value();
    }
    if (options.Has("mode")) {
      mode_ = options.Get("mode").As<Napi::Number>().Int32Value();
    }
  }
  
  // Configure AEC3 based on mode (0-7)
  EchoCanceller3Config config;
  config.filter.export_linear_aec_output = false;
  
  // Use HP filter only for aggressive modes
  use_hp_filter_ = (mode_ >= 6);
  
  switch (mode_) {
    case 0: // Ultra conservative
      config.suppressor.normal_tuning.mask_lf.enr_transparent = 0.8f;
      config.suppressor.normal_tuning.mask_lf.enr_suppress = 0.9f;
      config.suppressor.dominant_nearend_detection.enr_threshold = 0.1f;
      config.suppressor.dominant_nearend_detection.hold_duration = 100;
      break;
    case 1: // Conservative
      config.suppressor.normal_tuning.mask_lf.enr_transparent = 0.6f;
      config.suppressor.normal_tuning.mask_lf.enr_suppress = 0.75f;
      config.suppressor.dominant_nearend_detection.enr_threshold = 0.12f;
      config.suppressor.dominant_nearend_detection.hold_duration = 85;
      break;
    case 2: // Mild
      config.suppressor.normal_tuning.mask_lf.enr_transparent = 0.5f;
      config.suppressor.normal_tuning.mask_lf.enr_suppress = 0.6f;
      config.suppressor.dominant_nearend_detection.enr_threshold = 0.15f;
      config.suppressor.dominant_nearend_detection.hold_duration = 70;
      break;
    case 3: // Balanced
      config.suppressor.normal_tuning.mask_lf.enr_transparent = 0.4f;
      config.suppressor.normal_tuning.mask_lf.enr_suppress = 0.5f;
      config.suppressor.dominant_nearend_detection.enr_threshold = 0.2f;
      config.suppressor.dominant_nearend_detection.hold_duration = 60;
      break;
    case 4: // Moderate
      config.suppressor.normal_tuning.mask_lf.enr_transparent = 0.35f;
      config.suppressor.normal_tuning.mask_lf.enr_suppress = 0.45f;
      config.suppressor.dominant_nearend_detection.enr_threshold = 0.22f;
      config.suppressor.dominant_nearend_detection.hold_duration = 55;
      break;
    case 5: // Strong
      config.suppressor.normal_tuning.mask_lf.enr_transparent = 0.32f;
      config.suppressor.normal_tuning.mask_lf.enr_suppress = 0.42f;
      config.suppressor.dominant_nearend_detection.enr_threshold = 0.24f;
      config.suppressor.dominant_nearend_detection.hold_duration = 52;
      break;
    case 6: // Aggressive
      config.suppressor.normal_tuning.mask_lf.enr_transparent = 0.3f;
      config.suppressor.normal_tuning.mask_lf.enr_suppress = 0.4f;
      config.suppressor.dominant_nearend_detection.enr_threshold = 0.25f;
      config.suppressor.dominant_nearend_detection.hold_duration = 50;
      break;
    case 7: // Ultra aggressive (default WebRTC)
    default:
      // Use defaults
      break;
  }
  
  // Create AEC3
  EchoCanceller3Factory factory(config);
  echo_controller_ = factory.Create(sample_rate_, num_channels_, num_channels_);
  
  // Create HP filter if needed
  if (use_hp_filter_) {
    hp_filter_ = std::make_unique<HighPassFilter>(sample_rate_, num_channels_);
  }
  
  // Create audio buffers
  StreamConfig stream_config(sample_rate_, num_channels_, false);
  capture_buffer_ = std::make_unique<AudioBuffer>(
      stream_config.sample_rate_hz(), stream_config.num_channels(),
      stream_config.sample_rate_hz(), stream_config.num_channels(),
      stream_config.sample_rate_hz(), stream_config.num_channels());
  render_buffer_ = std::make_unique<AudioBuffer>(
      stream_config.sample_rate_hz(), stream_config.num_channels(),
      stream_config.sample_rate_hz(), stream_config.num_channels(),
      stream_config.sample_rate_hz(), stream_config.num_channels());
}

AEC3Wrapper::~AEC3Wrapper() {
  echo_controller_.reset();
  hp_filter_.reset();
  capture_buffer_.reset();
  render_buffer_.reset();
}

Napi::Value AEC3Wrapper::Process(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (info.Length() < 2) {
    Napi::TypeError::New(env, "Expected 2 arguments: capture and render arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  
  Napi::Float32Array capture = info[0].As<Napi::Float32Array>();
  Napi::Float32Array render = info[1].As<Napi::Float32Array>();
  
  size_t num_samples = capture.ElementLength();
  
  // Copy to audio buffers
  float* capture_data = capture_buffer_->channels()[0];
  float* render_data = render_buffer_->channels()[0];
  
  for (size_t i = 0; i < num_samples && i < static_cast<size_t>(frame_size_); i++) {
    capture_data[i] = capture[i];
    render_data[i] = render[i];
  }
  
  // Process render (reference) signal
  render_buffer_->SplitIntoFrequencyBands();
  echo_controller_->AnalyzeRender(render_buffer_.get());
  render_buffer_->MergeFrequencyBands();
  
  // Process capture signal
  echo_controller_->AnalyzeCapture(capture_buffer_.get());
  capture_buffer_->SplitIntoFrequencyBands();
  
  if (hp_filter_) {
    hp_filter_->Process(capture_buffer_.get(), true);
  }
  
  echo_controller_->SetAudioBufferDelay(0);
  echo_controller_->ProcessCapture(capture_buffer_.get(), nullptr, false);
  capture_buffer_->MergeFrequencyBands();
  
  // Create output array
  Napi::Float32Array output = Napi::Float32Array::New(env, num_samples);
  float* output_data = capture_buffer_->channels()[0];
  
  for (size_t i = 0; i < num_samples; i++) {
    output[i] = output_data[i];
  }
  
  return output;
}

Napi::Value AEC3Wrapper::AnalyzeRender(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "Expected render array argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  
  Napi::Float32Array render = info[0].As<Napi::Float32Array>();
  size_t num_samples = render.ElementLength();
  
  float* render_data = render_buffer_->channels()[0];
  for (size_t i = 0; i < num_samples && i < static_cast<size_t>(frame_size_); i++) {
    render_data[i] = render[i];
  }
  
  render_buffer_->SplitIntoFrequencyBands();
  echo_controller_->AnalyzeRender(render_buffer_.get());
  render_buffer_->MergeFrequencyBands();
  
  return env.Undefined();
}

Napi::Value AEC3Wrapper::ProcessCapture(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (info.Length() < 1) {
    Napi::TypeError::New(env, "Expected capture array argument")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  
  Napi::Float32Array capture = info[0].As<Napi::Float32Array>();
  size_t num_samples = capture.ElementLength();
  
  float* capture_data = capture_buffer_->channels()[0];
  for (size_t i = 0; i < num_samples && i < static_cast<size_t>(frame_size_); i++) {
    capture_data[i] = capture[i];
  }
  
  echo_controller_->AnalyzeCapture(capture_buffer_.get());
  capture_buffer_->SplitIntoFrequencyBands();
  
  if (hp_filter_) {
    hp_filter_->Process(capture_buffer_.get(), true);
  }
  
  echo_controller_->SetAudioBufferDelay(0);
  echo_controller_->ProcessCapture(capture_buffer_.get(), nullptr, false);
  capture_buffer_->MergeFrequencyBands();
  
  Napi::Float32Array output = Napi::Float32Array::New(env, num_samples);
  float* output_data = capture_buffer_->channels()[0];
  
  for (size_t i = 0; i < num_samples; i++) {
    output[i] = output_data[i];
  }
  
  return output;
}

Napi::Value AEC3Wrapper::Reset(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  // Recreate AEC3 to reset state
  EchoCanceller3Config config;
  EchoCanceller3Factory factory(config);
  echo_controller_ = factory.Create(sample_rate_, num_channels_, num_channels_);
  
  return env.Undefined();
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  return AEC3Wrapper::Init(env, exports);
}

NODE_API_MODULE(aec3_native, Init)

