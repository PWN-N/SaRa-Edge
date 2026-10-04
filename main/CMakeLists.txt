idf_component_register(
    SRCS "Main.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_i2s esp_driver_gpio
)