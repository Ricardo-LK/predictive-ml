package com.predictivemaintenance.web_service.client;

/** The Python inference service answered with an error or an unusable body. */
public class MLServiceException extends RuntimeException {
    public MLServiceException(String message, Throwable cause) {
        super(message, cause);
    }
}
