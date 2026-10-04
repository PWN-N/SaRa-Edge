#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "driver/i2s.h"

static const char *TAG = "SaRa-Edge";

// ============================================================
// INMP441 CONNECTIONS
// ============================================================

#define I2S_PORT        I2S_NUM_0

#define I2S_BCLK_PIN    GPIO_NUM_4
#define I2S_WS_PIN      GPIO_NUM_5
#define I2S_DATA_PIN    GPIO_NUM_6

// ============================================================
// AUDIO CONFIGURATION
// ============================================================

#define SAMPLE_RATE     16000

// 2 seconds of audio
#define AUDIO_SECONDS   2
#define AUDIO_SAMPLES   (SAMPLE_RATE * AUDIO_SECONDS)

// Buffer used for reading from I2S
#define I2S_READ_SAMPLES 1024

// ============================================================
// SETUP I2S
// ============================================================

static void init_i2s()
{
    i2s_config_t i2s_config = {};

    i2s_config.mode =
        (i2s_mode_t)(
            I2S_MODE_MASTER |
            I2S_MODE_RX
        );

    i2s_config.sample_rate = SAMPLE_RATE;

    // INMP441 sends 24-bit audio.
    // ESP32 I2S commonly receives it in 32-bit slots.
    i2s_config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;

    i2s_config.channel_format =
        I2S_CHANNEL_FMT_ONLY_LEFT;

    i2s_config.communication_format =
        I2S_COMM_FORMAT_I2S;

    i2s_config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;

    i2s_config.dma_buf_count = 8;

    i2s_config.dma_buf_len = 1024;

    i2s_config.use_apll = false;

    i2s_config.tx_desc_auto_clear = false;

    i2s_config.fixed_mclk = 0;


    // --------------------------------------------------------
    // Install I2S driver
    // --------------------------------------------------------

    ESP_ERROR_CHECK(
        i2s_driver_install(
            I2S_PORT,
            &i2s_config,
            0,
            NULL
        )
    );


    // --------------------------------------------------------
    // Configure GPIOs
    // --------------------------------------------------------

    i2s_pin_config_t pin_config = {};

    pin_config.bck_io_num = I2S_BCLK_PIN;

    pin_config.ws_io_num = I2S_WS_PIN;

    pin_config.data_out_num = I2S_PIN_NO_CHANGE;

    pin_config.data_in_num = I2S_DATA_PIN;


    ESP_ERROR_CHECK(
        i2s_set_pin(
            I2S_PORT,
            &pin_config
        )
    );


    // Clear DMA buffer
    i2s_zero_dma_buffer(I2S_PORT);

    ESP_LOGI(TAG, "I2S microphone initialized");
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
        AUDIO_SECONDS
    );


    while (sample_index < AUDIO_SAMPLES)
    {
        // ----------------------------------------------------
        // Read I2S data
        // ----------------------------------------------------

        ESP_ERROR_CHECK(
            i2s_read(
                I2S_PORT,
                raw_buffer,
                sizeof(raw_buffer),
                &bytes_read,
                portMAX_DELAY
            )
        );


        int samples_read =
            bytes_read / sizeof(int32_t);


        // ----------------------------------------------------
        // Convert INMP441 32-bit data to int16
        // ----------------------------------------------------

        for (int i = 0;
             i < samples_read &&
             sample_index < AUDIO_SAMPLES;
             i++)
        {
            int32_t sample = raw_buffer[i];

            // INMP441 audio is normally stored
            // in the upper portion of the 32-bit word.
            sample = sample >> 8;

            // Convert 24-bit-ish value to 16-bit.
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
        sample_index
    );
}

// ============================================================
// MAIN
// ============================================================

extern "C"
void app_main(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "          SaRa-Edge");
    ESP_LOGI(TAG, "       INMP441 Microphone");
    ESP_LOGI(TAG, "========================================");

    ESP_LOGI(
        TAG,
        "Sample rate: %d Hz",
        SAMPLE_RATE
    );

    ESP_LOGI(
        TAG,
        "Audio duration: %d seconds",
        AUDIO_SECONDS
    );

    ESP_LOGI(
        TAG,
        "Audio samples: %d",
        AUDIO_SAMPLES
    );


    // --------------------------------------------------------
    // Initialize microphone
    // --------------------------------------------------------

    init_i2s();


    // --------------------------------------------------------
    // Allocate 2-second audio buffer
    // --------------------------------------------------------

    int16_t *audio_buffer =
        (int16_t *)malloc(
            AUDIO_SAMPLES * sizeof(int16_t)
        );


    if (audio_buffer == NULL)
    {
        ESP_LOGE(
            TAG,
            "Failed to allocate audio buffer!"
        );

        return;
    }


    ESP_LOGI(
        TAG,
        "Audio buffer allocated: %d bytes",
        AUDIO_SAMPLES * sizeof(int16_t)
    );


    // --------------------------------------------------------
    // Continuous testing
    // --------------------------------------------------------

    while (true)
    {
        capture_audio(audio_buffer);


        // ----------------------------------------------------
        // Calculate simple audio level
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
                max_value = value;
        }


        float average =
            (float)sum /
            AUDIO_SAMPLES;


        ESP_LOGI(
            TAG,
            "Average amplitude: %.2f",
            average
        );

        ESP_LOGI(
            TAG,
            "Maximum amplitude: %d",
            max_value
        );


        // ----------------------------------------------------
        // Print first few samples
        // ----------------------------------------------------

        ESP_LOGI(
            TAG,
            "First 10 samples:"
        );

        for (int i = 0; i < 10; i++)
        {
            ESP_LOGI(
                TAG,
                "%d",
                audio_buffer[i]
            );
        }


        ESP_LOGI(
            TAG,
            "========================================"
        );


        vTaskDelay(
            pdMS_TO_TICKS(1000)
        );
    }
}