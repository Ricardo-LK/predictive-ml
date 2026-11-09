package com.predictivemaintenance.web_service.dto;

public record PredictionResponse (
    Integer prediction,
    Double failureProbability,
    Double threshold
) {}
