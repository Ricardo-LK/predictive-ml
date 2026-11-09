package com.predictivemaintenance.web_service.controller;

import com.predictivemaintenance.web_service.client.MLServiceException;
import com.predictivemaintenance.web_service.client.MLServiceUnavailableException;

import java.util.Map;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ProblemDetail;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/** Maps validation and upstream failures to RFC 9457 problem responses. */
@RestControllerAdvice
public class ApiExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(ApiExceptionHandler.class);

    @ExceptionHandler(MethodArgumentNotValidException.class)
    ProblemDetail handleValidation(MethodArgumentNotValidException e) {
        Map<String, String> errors = e.getBindingResult().getFieldErrors().stream()
            .collect(Collectors.toMap(
                f -> f.getField(),
                f -> String.valueOf(f.getDefaultMessage()),
                (a, b) -> a + "; " + b));

        ProblemDetail pd = ProblemDetail.forStatusAndDetail(HttpStatus.BAD_REQUEST, "Invalid prediction request");
        pd.setProperty("errors", errors);
        return pd;
    }

    @ExceptionHandler(MLServiceUnavailableException.class)
    ProblemDetail handleUnavailable(MLServiceUnavailableException e) {
        log.warn("{}: {}", e.getMessage(), e.getCause() != null ? e.getCause().getMessage() : "");
        return ProblemDetail.forStatusAndDetail(HttpStatus.SERVICE_UNAVAILABLE, e.getMessage());
    }

    @ExceptionHandler(MLServiceException.class)
    ProblemDetail handleUpstreamError(MLServiceException e) {
        log.error(e.getMessage(), e);
        return ProblemDetail.forStatusAndDetail(HttpStatus.BAD_GATEWAY, e.getMessage());
    }
}
