package com.predictivemaintenance.web_service.controller;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.predictivemaintenance.web_service.client.MLServiceException;
import com.predictivemaintenance.web_service.client.MLServiceUnavailableException;
import com.predictivemaintenance.web_service.dto.PredictionRequest;
import com.predictivemaintenance.web_service.dto.PredictionResponse;
import com.predictivemaintenance.web_service.service.PredictionService;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(PredictionController.class)
class PredictionControllerTest {

    private static final String VALID = """
        {"type":"L","airTemperature":298.1,"processTemperature":308.6,
         "rotationalSpeed":1551,"torque":42.8,"toolWear":0}
        """;

    @Autowired
    MockMvc mvc;

    @MockitoBean
    PredictionService service;

    @Test
    void forwardsValidRequestAndReturnsPrediction() throws Exception {
        PredictionRequest expected = new PredictionRequest("L", 298.1, 308.6, 1551, 42.8, 0);
        when(service.predict(expected)).thenReturn(new PredictionResponse(1, 0.73, 0.5));

        mvc.perform(post("/api/predictions").contentType(MediaType.APPLICATION_JSON).content(VALID))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.prediction").value(1))
            .andExpect(jsonPath("$.failureProbability").value(0.73))
            .andExpect(jsonPath("$.threshold").value(0.5));
    }

    @Test
    void rejectsMissingFieldWithoutCallingService() throws Exception {
        String body = VALID.replace("\"torque\":42.8,", "");

        mvc.perform(post("/api/predictions").contentType(MediaType.APPLICATION_JSON).content(body))
            .andExpect(status().isBadRequest())
            .andExpect(jsonPath("$.errors.torque").exists());

        verifyNoInteractions(service);
    }

    @Test
    void rejectsUnknownProductTypeAndNegativeValues() throws Exception {
        String body = VALID.replace("\"L\"", "\"X\"").replace("\"toolWear\":0", "\"toolWear\":-3");

        mvc.perform(post("/api/predictions").contentType(MediaType.APPLICATION_JSON).content(body))
            .andExpect(status().isBadRequest())
            .andExpect(jsonPath("$.errors.type").exists())
            .andExpect(jsonPath("$.errors.toolWear").exists());

        verifyNoInteractions(service);
    }

    @Test
    void ignoresUnknownFields() throws Exception {
        when(service.predict(any())).thenReturn(new PredictionResponse(0, 0.1, 0.5));
        String body = VALID.replace("\"toolWear\":0", "\"toolWear\":0,\"extra\":1");

        mvc.perform(post("/api/predictions").contentType(MediaType.APPLICATION_JSON).content(body))
            .andExpect(status().isOk());
    }

    @Test
    void rejectsMalformedJson() throws Exception {
        mvc.perform(post("/api/predictions").contentType(MediaType.APPLICATION_JSON).content("{oops"))
            .andExpect(status().isBadRequest());

        verifyNoInteractions(service);
    }

    @Test
    void returns503WhenInferenceServiceIsUnreachable() throws Exception {
        when(service.predict(any())).thenThrow(
            new MLServiceUnavailableException("ML inference service is unreachable", null));

        mvc.perform(post("/api/predictions").contentType(MediaType.APPLICATION_JSON).content(VALID))
            .andExpect(status().isServiceUnavailable())
            .andExpect(jsonPath("$.detail").value("ML inference service is unreachable"));
    }

    @Test
    void returns502WhenInferenceServiceFails() throws Exception {
        when(service.predict(any())).thenThrow(
            new MLServiceException("ML inference service returned HTTP 500", null));

        mvc.perform(post("/api/predictions").contentType(MediaType.APPLICATION_JSON).content(VALID))
            .andExpect(status().isBadGateway());
    }
}
