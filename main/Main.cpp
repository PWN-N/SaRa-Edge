#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_check.h"

// ============================================================
// ESP-DSP
// ============================================================

#include "dsps_fft2r.h"
#include "dsps_dct.h"

// ============================================================
// TFLITE MICRO
// ============================================================

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "kws_model.h"

// ============================================================
// TAG
// ============================================================

static const char *TAG = "SaRa-Edge";

// ============================================================
// INMP441 CONNECTIONS
// ============================================================

#define I2S_PORT I2S_NUM_0
static i2s_chan_handle_t rx_handle = NULL;

#define I2S_BCLK_PIN GPIO_NUM_4
#define I2S_WS_PIN GPIO_NUM_5
#define I2S_DATA_PIN GPIO_NUM_6

// ============================================================
// AUDIO CONFIGURATION
// ============================================================

#define SAMPLE_RATE 16000

// 2 seconds of audio
#define AUDIO_SECONDS 2
#define AUDIO_SAMPLES (SAMPLE_RATE * AUDIO_SECONDS)

// Buffer used for reading from I2S
#define I2S_READ_SAMPLES 1024

// ============================================================
// TFLITE CONFIGURATION
// ============================================================

// Temporary arena for first measurement.
// We will reduce this after measuring arena_used_bytes().
#define TENSOR_ARENA_SIZE (128 * 1024)

alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;

// ============================================================
// MFCC CONFIGURATION
// ============================================================
//
// These settings match the Python/librosa feature extraction:
//
// sample rate       = 16000
// n_fft             = 2048
// hop_length        = 512
// win_length        = 2048
// window            = periodic Hann
// center            = True
// pad_mode          = constant
// power             = 2
// n_mels            = 128
// fmin              = 0
// fmax              = 8000
// mel_norm          = Slaney
// n_mfcc            = 13
// DCT                = type-II, norm="ortho"
// power_to_db       = ref=1.0, top_db=80
//
// 2 seconds of 16 kHz audio produces 63 frames.
//

#define MFCC_N_FFT       2048
#define MFCC_HOP_LENGTH  512
#define MFCC_N_MELS      128
#define MFCC_N_MFCC      13
#define MFCC_PAD         (MFCC_N_FFT / 2)
#define MFCC_N_FRAMES    63

// librosa.load() converts integer WAV samples to approximately
// [-1, 1]. Our captured microphone samples are int16.
#define AUDIO_SCALE      (1.0f / 32768.0f)

// Model input quantization scale.
#define INPUT_SCALE      0.05341823026537895f

// Training-set normalization parameters.
static const float mfcc_mean[MFCC_N_MFCC] = {
    -642.85938f,
      23.249342f,
       3.0335546f,
       7.1449795f,
       2.6827378f,
       1.2719345f,
      -0.7957507f,
      -2.8949001f,
      -0.80792785f,
      -0.6110753f,
      -1.5597951f,
      -2.3444543f,
      -3.6013927f
};

static const float mfcc_std[MFCC_N_MFCC] = {
    224.7378f,
     57.648445f,
     25.272194f,
     24.34228f,
     14.292587f,
     12.280985f,
     11.782432f,
     10.840611f,
      7.834504f,
     10.101345f,
      7.0411067f,
      7.0537934f,
      8.421222f
};

// ============================================================
// MFCC WORK BUFFERS
// ============================================================
//
// FFT buffer:
//
// Re[0], Im[0], Re[1], Im[1], ... Re[N-1], Im[N-1]
//
// 2048 complex values = 4096 floats = 16 KB
//

alignas(16) static float fft_buffer[MFCC_N_FFT * 2];

// Periodic Hann window.
// 2048 floats = 8 KB.
static float hann_window[MFCC_N_FFT];

// 128 Mel filters require 130 boundary frequencies.
static float mel_frequencies[MFCC_N_MELS + 2];

static bool mfcc_initialized = false;

// ============================================================
// SETUP I2S
// ============================================================

static void init_i2s()
{
    i2s_chan_config_t chan_config =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_PORT,
            I2S_ROLE_MASTER);

    ESP_ERROR_CHECK(
        i2s_new_channel(
            &chan_config,
            NULL,
            &rx_handle));

    i2s_std_config_t std_config = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),

        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_32BIT,
            I2S_SLOT_MODE_MONO),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_PIN,
            .ws = I2S_WS_PIN,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_DATA_PIN,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false}}};

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(
            rx_handle,
            &std_config));

    ESP_ERROR_CHECK(
        i2s_channel_enable(rx_handle));

    ESP_LOGI(TAG, "I2S microphone interface initialized");
}

// ============================================================
// SETUP TFLITE MICRO
// ============================================================

static void init_tflite()
{
    ESP_LOGI(TAG, "Initializing TensorFlow Lite Micro...");

    // Load model from the embedded model array.
    model = tflite::GetModel(kws_model);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(
            TAG,
            "Model schema version %lu does not match supported version %d",
            static_cast<unsigned long>(model->version()),
            TFLITE_SCHEMA_VERSION);

        abort();
    }

    // Register only the operators used by our model.
    static tflite::MicroMutableOpResolver<6> resolver;

    if (resolver.AddConv2D() != kTfLiteOk)
        abort();

    if (resolver.AddMaxPool2D() != kTfLiteOk)
        abort();

    if (resolver.AddDepthwiseConv2D() != kTfLiteOk)
        abort();

    if (resolver.AddMean() != kTfLiteOk)
        abort();

    if (resolver.AddFullyConnected() != kTfLiteOk)
        abort();

    if (resolver.AddSoftmax() != kTfLiteOk)
        abort();

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE);

    interpreter = &static_interpreter;

    TfLiteStatus allocate_status =
        interpreter->AllocateTensors();

    if (allocate_status != kTfLiteOk)
    {
        ESP_LOGE(TAG, "AllocateTensors() failed!");
        abort();
    }

    TfLiteTensor *input = interpreter->input(0);
    TfLiteTensor *output = interpreter->output(0);

    ESP_LOGI(
        TAG,
        "Tensor arena: %u bytes allocated",
        static_cast<unsigned>(interpreter->arena_used_bytes()));

    ESP_LOGI(
        TAG,
        "Tensor arena capacity: %u bytes",
        static_cast<unsigned>(TENSOR_ARENA_SIZE));

    ESP_LOGI(
        TAG,
        "Input type: %d, shape: [%d, %d, %d, %d]",
        input->type,
        input->dims->data[0],
        input->dims->data[1],
        input->dims->data[2],
        input->dims->data[3]);

    ESP_LOGI(
        TAG,
        "Output type: %d, shape: [%d, %d]",
        output->type,
        output->dims->data[0],
        output->dims->data[1]);

    size_t free_heap =
        heap_caps_get_free_size(MALLOC_CAP_8BIT);

    size_t largest_block =
        heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);

    ESP_LOGI(
        TAG,
        "Free 8-bit heap: %u bytes",
        static_cast<unsigned>(free_heap));

    ESP_LOGI(
        TAG,
        "Largest 8-bit heap block: %u bytes",
        static_cast<unsigned>(largest_block));
}

// ============================================================
// LIBROSA / SLANEY MEL SCALE
// ============================================================

static float hz_to_mel_slaney(float hz)
{
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = logf(6.4f) / 27.0f;

    if (hz < min_log_hz)
    {
        return hz / f_sp;
    }

    return min_log_mel +
           logf(hz / min_log_hz) / logstep;
}

static float mel_to_hz_slaney(float mel)
{
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = logf(6.4f) / 27.0f;

    if (mel < min_log_mel)
    {
        return f_sp * mel;
    }

    return min_log_hz *
           expf(logstep * (mel - min_log_mel));
}

// ============================================================
// INITIALIZE MFCC
// ============================================================

static void init_mfcc()
{
    ESP_LOGI(TAG, "Initializing MFCC pipeline...");

    // ========================================================
    // Periodic Hann window
    // ========================================================
    //
    // librosa/scipy uses the periodic form:
    //
    // w[n] = 0.5 * (1 - cos(2*pi*n/N))
    //
    // ESP-DSP's generic Hann helper uses N-1, so we
    // generate the required window ourselves.
    //

    for (int n = 0; n < MFCC_N_FFT; n++)
    {
        hann_window[n] =
            0.5f *
            (1.0f -
             cosf(
                 2.0f *
                 3.14159265358979323846f *
                 (float)n /
                 (float)MFCC_N_FFT));
    }

    // ========================================================
    // Slaney Mel frequencies
    // ========================================================

    const float min_mel =
        hz_to_mel_slaney(0.0f);

    const float max_mel =
        hz_to_mel_slaney(8000.0f);

    for (int i = 0;
         i < MFCC_N_MELS + 2;
         i++)
    {
        float mel =
            min_mel +
            ((max_mel - min_mel) *
             (float)i /
             (float)(MFCC_N_MELS + 1));

        mel_frequencies[i] =
            mel_to_hz_slaney(mel);
    }

    // ========================================================
    // Initialize ESP-DSP FFT
    // ========================================================

    esp_err_t ret =
        dsps_fft2r_init_fc32(
            NULL,
            MFCC_N_FFT);

    if (ret != ESP_OK)
    {
        ESP_LOGE(
            TAG,
            "FFT initialization failed: %s",
            esp_err_to_name(ret));

        return;
    }

    mfcc_initialized = true;

    ESP_LOGI(
        TAG,
        "MFCC initialized: FFT=%d, hop=%d, mel=%d, frames=%d",
        MFCC_N_FFT,
        MFCC_HOP_LENGTH,
        MFCC_N_MELS,
        MFCC_N_FRAMES);
}

// ============================================================
// CALCULATE ALL MEL FILTER ENERGIES FOR ONE FRAME
// ============================================================
//
// The FFT is calculated ONCE per frame.
//
// One FFT
//    |
//    +--> Mel 0 energy
//    +--> Mel 1 energy
//    +--> ...
//    +--> Mel 127 energy
//
// This avoids calculating a separate 2048-point FFT for
// every Mel filter.
//

static void calculate_mel_energies(
    const int16_t *audio_buffer,
    int frame,
    float *mel_energies)
{
    // Centered frame.
    //
    // For frame 0:
    //
    // start = -1024
    //
    // This matches librosa center=True with
    // pad_mode="constant".

    const int frame_start =
        frame * MFCC_HOP_LENGTH -
        MFCC_PAD;

    const float bin_frequency =
        (float)SAMPLE_RATE /
        (float)MFCC_N_FFT;

    // ========================================================
    // Build centered + windowed FFT frame
    // ========================================================

    for (int n = 0;
         n < MFCC_N_FFT;
         n++)
    {
        const int audio_index =
            frame_start + n;

        float sample = 0.0f;

        if (audio_index >= 0 &&
            audio_index < AUDIO_SAMPLES)
        {
            sample =
                (float)audio_buffer[audio_index] *
                AUDIO_SCALE;
        }

        fft_buffer[2 * n] =
            sample * hann_window[n];

        fft_buffer[2 * n + 1] =
            0.0f;
    }

    // ========================================================
    // FFT
    // ========================================================

    dsps_fft2r_fc32(
        fft_buffer,
        MFCC_N_FFT);

    dsps_bit_rev_fc32(
        fft_buffer,
        MFCC_N_FFT);

    // ========================================================
    // Calculate all 128 Mel filter energies
    // ========================================================

    for (int mel = 0;
         mel < MFCC_N_MELS;
         mel++)
    {
        const float f0 =
            mel_frequencies[mel];

        const float f1 =
            mel_frequencies[mel + 1];

        const float f2 =
            mel_frequencies[mel + 2];

        const float left_denominator =
            f1 - f0;

        const float right_denominator =
            f2 - f1;

        // librosa Slaney normalization:
        //
        // 2 / (mel_f[i+2] - mel_f[i])
        const float slaney_norm =
            2.0f /
            (f2 - f0);

        int start_bin =
            (int)ceilf(
                f0 / bin_frequency);

        int end_bin =
            (int)floorf(
                f2 / bin_frequency);

        if (start_bin < 0)
            start_bin = 0;

        if (end_bin > MFCC_N_FFT / 2)
            end_bin = MFCC_N_FFT / 2;

        float energy = 0.0f;

        for (int k = start_bin;
             k <= end_bin;
             k++)
        {
            const float re =
                fft_buffer[2 * k];

            const float im =
                fft_buffer[2 * k + 1];

            // librosa power=2:
            //
            // |STFT|² = Re² + Im²
            const float power =
                re * re +
                im * im;

            const float frequency =
                (float)k *
                bin_frequency;

            float lower =
                (frequency - f0) /
                left_denominator;

            float upper =
                (f2 - frequency) /
                right_denominator;

            float weight =
                fminf(
                    lower,
                    upper);

            if (weight < 0.0f)
                weight = 0.0f;

            energy +=
                power *
                weight *
                slaney_norm;
        }

        mel_energies[mel] =
            energy;
    }
}

// ============================================================
// MFCC EXTRACTION + MODEL INPUT QUANTIZATION
// ============================================================

static bool extract_mfcc_and_quantize(
    const int16_t *audio_buffer)
{
    if (!mfcc_initialized)
    {
        ESP_LOGE(
            TAG,
            "MFCC is not initialized");

        return false;
    }

    if (interpreter == nullptr)
    {
        ESP_LOGE(
            TAG,
            "TFLite interpreter is not initialized");

        return false;
    }

    TfLiteTensor *input =
        interpreter->input(0);

    if (input == nullptr)
    {
        ESP_LOGE(
            TAG,
            "Input tensor is null");

        return false;
    }

    if (input->type != kTfLiteInt8)
    {
        ESP_LOGE(
            TAG,
            "Expected int8 input tensor");

        return false;
    }

    // ========================================================
    // Temporary Mel-energy buffer
    // ========================================================
    //
    // 128 floats = 512 bytes.
    //
    // Reused for every frame.
    //

    float mel_energies[MFCC_N_MELS];

    // ========================================================
    // PASS 1
    // ========================================================
    //
    // librosa.power_to_db(top_db=80) uses the maximum value
    // of the complete Mel spectrogram.
    //
    // Therefore we first determine the global maximum.
    //

    float global_max_power =
        1e-10f;

    for (int frame = 0;
         frame < MFCC_N_FRAMES;
         frame++)
    {
        calculate_mel_energies(
            audio_buffer,
            frame,
            mel_energies);

        for (int mel = 0;
             mel < MFCC_N_MELS;
             mel++)
        {
            if (mel_energies[mel] >
                global_max_power)
            {
                global_max_power =
                    mel_energies[mel];
            }
        }
    }

    const float global_max_db =
        10.0f *
        log10f(
            fmaxf(
                global_max_power,
                1e-10f));

    const float floor_db =
        global_max_db - 80.0f;

    // ========================================================
    // PASS 2
    // ========================================================

    for (int frame = 0;
         frame < MFCC_N_FRAMES;
         frame++)
    {
        // One FFT calculates all 128 Mel energies.
        calculate_mel_energies(
            audio_buffer,
            frame,
            mel_energies);

        // ----------------------------------------------------
        // Mel power -> dB
        // ----------------------------------------------------

        for (int mel = 0;
             mel < MFCC_N_MELS;
             mel++)
        {
            const float safe_energy =
                fmaxf(
                    mel_energies[mel],
                    1e-10f);

            float db =
                10.0f *
                log10f(safe_energy);

            // librosa top_db=80
            if (db < floor_db)
                db = floor_db;

            // IMPORTANT:
            //
            // dsps_dct_f32() expects the 128 input values
            // consecutively in fft_buffer[0..127].
            fft_buffer[mel] =
                db;
        }

        // ----------------------------------------------------
        // DCT-II
        // ----------------------------------------------------

        dsps_dct_f32(
            fft_buffer,
            MFCC_N_MELS);

        // ----------------------------------------------------
        // Normalize + quantize
        // ----------------------------------------------------

        for (int mfcc_index = 0;
             mfcc_index < MFCC_N_MFCC;
             mfcc_index++)
        {
            float coefficient =
                fft_buffer[mfcc_index];

            // librosa DCT norm="ortho"
            if (mfcc_index == 0)
            {
                coefficient /=
                    sqrtf(
                        (float)MFCC_N_MELS);
            }
            else
            {
                coefficient *=
                    sqrtf(
                        2.0f /
                        (float)MFCC_N_MELS);
            }

            // normalization_v2.npz
            const float normalized =
                (coefficient -
                 mfcc_mean[mfcc_index]) /
                mfcc_std[mfcc_index];

            // Model input:
            //
            // scale = 0.05341823026537895
            // zero_point = 0
            //
            // q = round(x / scale)

            long quantized =
                lrintf(
                    normalized /
                    INPUT_SCALE);

            if (quantized > 127)
                quantized = 127;

            if (quantized < -128)
                quantized = -128;

            // Input shape:
            //
            // [1, 13, 63, 1]
            //
            // Row-major:
            //
            // [MFCC][FRAME]

            input->data.int8[
                mfcc_index *
                MFCC_N_FRAMES +
                frame
            ] = (int8_t)quantized;
        }
    }

    return true;
}

// ============================================================
// RUN TFLITE INFERENCE
// ============================================================

static void run_inference()
{
    ESP_LOGI(
        TAG,
        "Running inference...");

    size_t free_before =
        heap_caps_get_free_size(
            MALLOC_CAP_8BIT);

    TfLiteStatus status =
        interpreter->Invoke();

    if (status != kTfLiteOk)
    {
        ESP_LOGE(
            TAG,
            "Invoke failed!");

        return;
    }

    size_t free_after =
        heap_caps_get_free_size(
            MALLOC_CAP_8BIT);

    TfLiteTensor *output =
        interpreter->output(0);

    if (output == nullptr)
    {
        ESP_LOGE(
            TAG,
            "Output tensor is null");

        return;
    }

    // --------------------------------------------------------
    // Argmax
    // --------------------------------------------------------

    int best_class = 0;

    for (int i = 1; i < 3; i++)
    {
        if (output->data.int8[i] >
            output->data.int8[best_class])
        {
            best_class = i;
        }
    }

    const char *class_names[] = {
        "keyword",
        "unknown",
        "noise"
    };

    ESP_LOGI(
        TAG,
        "Prediction: %s (%d)",
        class_names[best_class],
        best_class);

    // --------------------------------------------------------
    // Output scores
    // --------------------------------------------------------
    //
    // output scale = 0.00390625
    // output zero_point = -128
    //
    // float = (int8 - (-128)) * scale
    //

    for (int i = 0; i < 3; i++)
    {
        const float score =
            ((float)output->data.int8[i] +
             128.0f) *
            0.00390625f;

        ESP_LOGI(
            TAG,
            "class[%d] %s: int8=%d score=%.4f",
            i,
            class_names[i],
            output->data.int8[i],
            score);
    }

    // --------------------------------------------------------
    // Runtime heap measurement
    // --------------------------------------------------------

    ESP_LOGI(
        TAG,
        "Free heap before Invoke: %u bytes",
        static_cast<unsigned>(
            free_before));

    ESP_LOGI(
        TAG,
        "Free heap after Invoke: %u bytes",
        static_cast<unsigned>(
            free_after));

    ESP_LOGI(
        TAG,
        "Largest free block: %u bytes",
        static_cast<unsigned>(
            heap_caps_get_largest_free_block(
                MALLOC_CAP_8BIT)));
}

// ============================================================
// CAPTURE AUDIO
// ============================================================

static void capture_audio(int16_t *audio_buffer)
{
    size_t bytes_read = 0;

    int32_t raw_buffer[I2S_READ_SAMPLES];

    int sample_index = 0;

    ESP_LOGI(
        TAG,
        "Recording %d seconds...",
        AUDIO_SECONDS);

    while (sample_index < AUDIO_SAMPLES)
    {
        ESP_ERROR_CHECK(
            i2s_channel_read(
                rx_handle,
                raw_buffer,
                sizeof(raw_buffer),
                &bytes_read,
                portMAX_DELAY));

        int samples_read =
            bytes_read /
            sizeof(int32_t);

        for (int i = 0;
             i < samples_read &&
             sample_index < AUDIO_SAMPLES;
             i++)
        {
            int32_t sample =
                raw_buffer[i];

            sample = sample >> 8;
            sample = sample >> 8;

            if (sample > 32767)
                sample = 32767;

            if (sample < -32768)
                sample = -32768;

            audio_buffer[sample_index++] =
                (int16_t)sample;
        }
    }

    ESP_LOGI(
        TAG,
        "Recording complete: %d samples",
        sample_index);
}

// ============================================================
// MAIN
// ============================================================

extern "C" void app_main(void)
{
    ESP_LOGI(
        TAG,
        "========================================");

    ESP_LOGI(
        TAG,
        "              SaRa-Edge");

    ESP_LOGI(
        TAG,
        "          INMP441 Microphone");

    ESP_LOGI(
        TAG,
        "========================================");

    ESP_LOGI(
        TAG,
        "Sample rate: %d Hz",
        SAMPLE_RATE);

    ESP_LOGI(
        TAG,
        "Audio duration: %d seconds",
        AUDIO_SECONDS);

    ESP_LOGI(
        TAG,
        "Audio samples: %d",
        AUDIO_SAMPLES);

    // --------------------------------------------------------
    // I2S
    // --------------------------------------------------------

    init_i2s();

    // --------------------------------------------------------
    // Audio buffer
    // --------------------------------------------------------

    int16_t *audio_buffer =
        (int16_t *)malloc(
            AUDIO_SAMPLES *
            sizeof(int16_t));

    if (audio_buffer == NULL)
    {
        ESP_LOGE(
            TAG,
            "Failed to allocate audio buffer!");

        return;
    }

    ESP_LOGI(
        TAG,
        "Audio buffer allocated: %d bytes",
        AUDIO_SAMPLES *
        sizeof(int16_t));

    // --------------------------------------------------------
    // TFLite
    // --------------------------------------------------------

    init_tflite();

    // --------------------------------------------------------
    // MFCC
    // --------------------------------------------------------

    init_mfcc();

    if (!mfcc_initialized)
    {
        ESP_LOGE(
            TAG,
            "MFCC initialization failed!");

        free(audio_buffer);

        return;
    }

    // --------------------------------------------------------
    // Main inference loop
    // --------------------------------------------------------

    while (true)
    {
        ESP_LOGI(
            TAG,
            "========================================");

        // ----------------------------------------------------
        // Capture
        // ----------------------------------------------------

        capture_audio(
            audio_buffer);

        // ----------------------------------------------------
        // Audio amplitude information
        // ----------------------------------------------------

        int64_t sum = 0;
        int16_t max_value = 0;

        for (int i = 0;
             i < AUDIO_SAMPLES;
             i++)
        {
            int32_t value =
                audio_buffer[i];

            if (value < 0)
                value = -value;

            sum += value;

            if (value > max_value)
                max_value =
                    (int16_t)value;
        }

        float average =
            (float)sum /
            AUDIO_SAMPLES;

        ESP_LOGI(
            TAG,
            "Average amplitude: %.2f",
            average);

        ESP_LOGI(
            TAG,
            "Maximum amplitude: %d",
            max_value);

        // ----------------------------------------------------
        // MFCC + quantization
        // ----------------------------------------------------

        if (extract_mfcc_and_quantize(
                audio_buffer))
        {
            ESP_LOGI(
                TAG,
                "MFCC extraction and quantization complete");

            // ------------------------------------------------
            // TFLite inference
            // ------------------------------------------------

            run_inference();
        }
        else
        {
            ESP_LOGE(
                TAG,
                "MFCC extraction failed!");
        }

        ESP_LOGI(
            TAG,
            "========================================");

        vTaskDelay(
            pdMS_TO_TICKS(1000));
    }
}