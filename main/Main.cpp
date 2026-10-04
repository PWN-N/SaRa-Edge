#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_check.h"
#include <stdlib.h>

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
            bytes_read / sizeof(int32_t);

        for (int i = 0;
             i < samples_read &&
             sample_index < AUDIO_SAMPLES;
             i++)
        {
            int32_t sample = raw_buffer[i];

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
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "          SaRa-Edge");
    ESP_LOGI(TAG, "       INMP441 Microphone");
    ESP_LOGI(TAG, "========================================");

    ESP_LOGI(TAG, "Sample rate: %d Hz", SAMPLE_RATE);
    ESP_LOGI(TAG, "Audio duration: %d seconds", AUDIO_SECONDS);
    ESP_LOGI(TAG, "Audio samples: %d", AUDIO_SAMPLES);

    init_i2s();

    int16_t *audio_buffer =
        (int16_t *)malloc(
            AUDIO_SAMPLES * sizeof(int16_t));

    if (audio_buffer == NULL)
    {
        ESP_LOGE(TAG, "Failed to allocate audio buffer!");
        return;
    }

    ESP_LOGI(
        TAG,
        "Audio buffer allocated: %d bytes",
        AUDIO_SAMPLES * sizeof(int16_t));

    init_tflite();

    while (true)
    {
        capture_audio(audio_buffer);

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
                max_value = value;
        }

        float average =
            (float)sum /
            AUDIO_SAMPLES;

        ESP_LOGI(TAG, "Average amplitude: %.2f", average);
        ESP_LOGI(TAG, "Maximum amplitude: %d", max_value);

        ESP_LOGI(TAG, "First 10 samples:");

        for (int i = 0; i < 10; i++)
        {
            ESP_LOGI(TAG, "%d", audio_buffer[i]);
        }

        ESP_LOGI(TAG, "========================================");

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
