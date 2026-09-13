package com.predictivemaintenance.web_service.client;

import com.predictivemaintenance.web_service.dto.PredictionRequest;
import com.predictivemaintenance.web_service.dto.PredictionResponse;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
public class MLClient {
    private final RestClient restClient;

    public MLClient( RestClient.Builder builder, @Value("${ml.service.url}") String MLServiceUrl ) {
        this.restClient = builder.baseUrl(MLServiceUrl).build();
    }

    public PredictionResponse predict (PredictionRequest req) {
        return restClient.post().uri("/predict")
        .body(req).retrieve().body(PredictionResponse.class);
    }
}
