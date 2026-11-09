package com.predictivemaintenance.web_service.client;

import com.predictivemaintenance.web_service.dto.PredictionRequest;
import com.predictivemaintenance.web_service.dto.PredictionResponse;

import java.net.http.HttpClient;
import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

@Component
public class MLClient {
    private final RestClient restClient;

    public MLClient(
        RestClient.Builder builder,
        @Value("${ml.service.url}") String MLServiceUrl,
        @Value("${ml.service.connect-timeout:2s}") Duration connectTimeout,
        @Value("${ml.service.read-timeout:5s}") Duration readTimeout
    ) {
        HttpClient httpClient = HttpClient.newBuilder().connectTimeout(connectTimeout).build();
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(readTimeout);

        this.restClient = builder.baseUrl(MLServiceUrl).requestFactory(requestFactory).build();
    }

    public PredictionResponse predict (PredictionRequest req) {
        PredictionResponse res;
        try {
            res = restClient.post().uri("/predict")
            .body(req).retrieve().body(PredictionResponse.class);
        } catch (ResourceAccessException e) {
            // Connection refused, DNS failure, connect/read timeout
            throw new MLServiceUnavailableException("ML inference service is unreachable", e);
        } catch (RestClientResponseException e) {
            throw new MLServiceException(
                "ML inference service returned HTTP " + e.getStatusCode().value(), e);
        } catch (RestClientException e) {
            throw new MLServiceException("Invalid response from ML inference service", e);
        }

        if (res == null || res.prediction() == null || res.failureProbability() == null
                || res.threshold() == null) {
            throw new MLServiceException("Incomplete response from ML inference service", null);
        }
        return res;
    }
}
