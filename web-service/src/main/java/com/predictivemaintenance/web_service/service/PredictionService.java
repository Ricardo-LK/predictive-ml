package com.predictivemaintenance.web_service.service;

import com.predictivemaintenance.web_service.client.MLClient;
import com.predictivemaintenance.web_service.dto.PredictionRequest;
import com.predictivemaintenance.web_service.dto.PredictionResponse;

import org.springframework.stereotype.Service;

@Service
public class PredictionService {
    private final MLClient client;

    public PredictionService(MLClient client) {
        this.client = client;
    }

    public PredictionResponse predict(PredictionRequest req) {
        return client.predict(req);
    }
}

