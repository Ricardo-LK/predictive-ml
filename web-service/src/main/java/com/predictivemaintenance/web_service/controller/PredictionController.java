package com.predictivemaintenance.web_service.controller;

import com.predictivemaintenance.web_service.dto.PredictionRequest;
import com.predictivemaintenance.web_service.dto.PredictionResponse;
import com.predictivemaintenance.web_service.service.PredictionService;

import jakarta.validation.Valid;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/predictions")
public class PredictionController {

    private final PredictionService service;

    public PredictionController (PredictionService service) {
        this.service = service;
    }

    @PostMapping
    public ResponseEntity<PredictionResponse> predict(@Valid @RequestBody PredictionRequest req) {
        PredictionResponse res = service.predict(req);

        return ResponseEntity.ok(res);
    }
}