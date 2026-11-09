package com.predictivemaintenance.web_service.dto;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.PositiveOrZero;

/**
 * Mirrors the Python API's PredictionRequest. Wrapper types plus @NotNull make
 * missing fields a 400 instead of silently defaulting to 0.
 */
public record PredictionRequest(

    @NotNull
    @Pattern(regexp = "[LMH]", message = "must be one of L, M, H")
    String type,

    /** Air temperature [K]. */
    @NotNull @Positive
    Double airTemperature,

    /** Process temperature [K]. */
    @NotNull @Positive
    Double processTemperature,

    /** Rotational speed [rpm]. */
    @NotNull @PositiveOrZero
    Integer rotationalSpeed,

    /** Torque [Nm]. */
    @NotNull @PositiveOrZero
    Double torque,

    /** Tool wear [min]. */
    @NotNull @PositiveOrZero
    Integer toolWear

) {}
