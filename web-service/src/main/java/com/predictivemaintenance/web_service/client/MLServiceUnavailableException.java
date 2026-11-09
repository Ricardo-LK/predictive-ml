package com.predictivemaintenance.web_service.client;

/** The Python inference service could not be reached or timed out. */
public class MLServiceUnavailableException extends RuntimeException {
    public MLServiceUnavailableException(String message, Throwable cause) {
        super(message, cause);
    }
}
