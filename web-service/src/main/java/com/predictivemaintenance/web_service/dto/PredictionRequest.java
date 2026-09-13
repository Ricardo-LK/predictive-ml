package com.predictivemaintenance.web_service.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.PositiveOrZero;

public record PredictionRequest(

    @NotBlank
    String type,

    @PositiveOrZero
    double airTemperature,

    @PositiveOrZero
    double processTemperature,

    @PositiveOrZero
    int rotationalSpeed,

    @PositiveOrZero
    double torque,

    @PositiveOrZero
    int toolWear

) {}