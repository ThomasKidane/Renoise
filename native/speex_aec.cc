/**
 * SpeexDSP Echo Cancellation Native Addon for Node.js
 * 
 * Uses SpeexDSP's acoustic echo cancellation (AEC) algorithm
 */

#include <napi.h>
#include <speex/speex_echo.h>
#include <speex/speex_preprocess.h>
#include <vector>
#include <cstring>

class SpeexAEC : public Napi::ObjectWrap<SpeexAEC> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    SpeexAEC(const Napi::CallbackInfo& info);
    ~SpeexAEC();

private:
    static Napi::FunctionReference constructor;
    
    Napi::Value Process(const Napi::CallbackInfo& info);
    Napi::Value Reset(const Napi::CallbackInfo& info);
    
    SpeexEchoState* echo_state;
    SpeexPreprocessState* preprocess_state;
    int frame_size;
    int sample_rate;
    int filter_length;
    
    std::vector<spx_int16_t> input_buffer;
    std::vector<spx_int16_t> ref_buffer;
    std::vector<spx_int16_t> output_buffer;
};

Napi::FunctionReference SpeexAEC::constructor;

Napi::Object SpeexAEC::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);

    Napi::Function func = DefineClass(env, "SpeexAEC", {
        InstanceMethod("process", &SpeexAEC::Process),
        InstanceMethod("reset", &SpeexAEC::Reset),
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("SpeexAEC", func);
    return exports;
}

SpeexAEC::SpeexAEC(const Napi::CallbackInfo& info) : Napi::ObjectWrap<SpeexAEC>(info) {
    Napi::Env env = info.Env();

    // Default parameters
    this->frame_size = 160;      // 10ms at 16kHz, or 20ms at 8kHz
    this->sample_rate = 16000;   // 16kHz default
    this->filter_length = 4096;  // ~256ms at 16kHz - good for room echo

    if (info.Length() > 0 && info[0].IsObject()) {
        Napi::Object options = info[0].As<Napi::Object>();
        
        if (options.Has("frameSize")) {
            this->frame_size = options.Get("frameSize").As<Napi::Number>().Int32Value();
        }
        if (options.Has("sampleRate")) {
            this->sample_rate = options.Get("sampleRate").As<Napi::Number>().Int32Value();
        }
        if (options.Has("filterLength")) {
            this->filter_length = options.Get("filterLength").As<Napi::Number>().Int32Value();
        }
    }

    // Initialize echo canceller
    this->echo_state = speex_echo_state_init(this->frame_size, this->filter_length);
    if (!this->echo_state) {
        Napi::Error::New(env, "Failed to initialize Speex echo state").ThrowAsJavaScriptException();
        return;
    }

    speex_echo_ctl(this->echo_state, SPEEX_ECHO_SET_SAMPLING_RATE, &this->sample_rate);

    // Initialize preprocessor for additional noise suppression
    this->preprocess_state = speex_preprocess_state_init(this->frame_size, this->sample_rate);
    if (this->preprocess_state) {
        speex_preprocess_ctl(this->preprocess_state, SPEEX_PREPROCESS_SET_ECHO_STATE, this->echo_state);
        
        // Enable noise suppression
        int denoise = 1;
        speex_preprocess_ctl(this->preprocess_state, SPEEX_PREPROCESS_SET_DENOISE, &denoise);
        
        // Set noise suppression level (-15 dB)
        int noise_suppress = -15;
        speex_preprocess_ctl(this->preprocess_state, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, &noise_suppress);
        
        // Enable residual echo suppression
        int echo_suppress = -40;
        speex_preprocess_ctl(this->preprocess_state, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS, &echo_suppress);
        int echo_suppress_active = -15;
        speex_preprocess_ctl(this->preprocess_state, SPEEX_PREPROCESS_SET_ECHO_SUPPRESS_ACTIVE, &echo_suppress_active);
    }

    // Allocate buffers
    this->input_buffer.resize(this->frame_size);
    this->ref_buffer.resize(this->frame_size);
    this->output_buffer.resize(this->frame_size);
}

SpeexAEC::~SpeexAEC() {
    if (this->echo_state) {
        speex_echo_state_destroy(this->echo_state);
        this->echo_state = nullptr;
    }
    if (this->preprocess_state) {
        speex_preprocess_state_destroy(this->preprocess_state);
        this->preprocess_state = nullptr;
    }
}

/**
 * Process audio through echo cancellation
 * 
 * @param input Float32Array - microphone input (with echo)
 * @param reference Float32Array - reference signal (what's being played)
 * @returns Float32Array - echo-cancelled output
 */
Napi::Value SpeexAEC::Process(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected 2 arguments: input and reference").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!info[0].IsTypedArray() || !info[1].IsTypedArray()) {
        Napi::TypeError::New(env, "Arguments must be Float32Array").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Float32Array input = info[0].As<Napi::Float32Array>();
    Napi::Float32Array reference = info[1].As<Napi::Float32Array>();

    size_t num_samples = input.ElementLength();
    if (reference.ElementLength() != num_samples) {
        Napi::TypeError::New(env, "Input and reference must have same length").ThrowAsJavaScriptException();
        return env.Null();
    }

    // Create output array
    Napi::Float32Array output = Napi::Float32Array::New(env, num_samples);

    // Process in frame_size chunks
    for (size_t offset = 0; offset + this->frame_size <= num_samples; offset += this->frame_size) {
        // Convert float32 [-1, 1] to int16 [-32768, 32767]
        for (int i = 0; i < this->frame_size; i++) {
            float in_sample = input[offset + i];
            float ref_sample = reference[offset + i];
            
            // Clamp and convert
            in_sample = std::max(-1.0f, std::min(1.0f, in_sample));
            ref_sample = std::max(-1.0f, std::min(1.0f, ref_sample));
            
            this->input_buffer[i] = static_cast<spx_int16_t>(in_sample * 32767.0f);
            this->ref_buffer[i] = static_cast<spx_int16_t>(ref_sample * 32767.0f);
        }

        // Run echo cancellation
        speex_echo_cancellation(
            this->echo_state,
            this->input_buffer.data(),
            this->ref_buffer.data(),
            this->output_buffer.data()
        );

        // Run preprocessor for additional noise suppression
        if (this->preprocess_state) {
            speex_preprocess_run(this->preprocess_state, this->output_buffer.data());
        }

        // Convert back to float32
        for (int i = 0; i < this->frame_size; i++) {
            output[offset + i] = static_cast<float>(this->output_buffer[i]) / 32767.0f;
        }
    }

    // Handle remaining samples (just pass through)
    size_t remaining_start = (num_samples / this->frame_size) * this->frame_size;
    for (size_t i = remaining_start; i < num_samples; i++) {
        output[i] = input[i];
    }

    return output;
}

Napi::Value SpeexAEC::Reset(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (this->echo_state) {
        speex_echo_state_reset(this->echo_state);
    }

    return env.Undefined();
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    return SpeexAEC::Init(env, exports);
}

NODE_API_MODULE(speex_aec, Init)


