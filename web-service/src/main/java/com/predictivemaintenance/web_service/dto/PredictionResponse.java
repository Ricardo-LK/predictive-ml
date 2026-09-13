package com.predictivemaintenance.web_service.dto;

public record PredictionResponse (
    int prediction,
    double failureProbability,
    double threshold
) {}
