package com.predictivemaintenance.web_service.client;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.predictivemaintenance.web_service.dto.PredictionRequest;
import com.predictivemaintenance.web_service.dto.PredictionResponse;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.web.client.RestClient;

/** Exercises MLClient against a real local HTTP server standing in for FastAPI. */
class MLClientTest {

    private static final PredictionRequest REQUEST =
        new PredictionRequest("M", 300.0, 310.5, 1400, 55.0, 180);

    private HttpServer server;

    @AfterEach
    void stop() {
        if (server != null) server.stop(0);
    }

    private String startServer(int status, String body, long delayMs, AtomicReference<String> received)
            throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/predict", exchange -> {
            received.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            try {
                Thread.sleep(delayMs);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
            byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(status, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    private MLClient client(String url) {
        return new MLClient(RestClient.builder(), url, Duration.ofMillis(500), Duration.ofMillis(500));
    }

    @Test
    void sendsCamelCaseContractAndParsesResponse() throws Exception {
        AtomicReference<String> received = new AtomicReference<>();
        String url = startServer(200,
            "{\"prediction\":0,\"failureProbability\":0.12,\"threshold\":0.5}", 0, received);

        PredictionResponse res = client(url).predict(REQUEST);

        assertThat(res).isEqualTo(new PredictionResponse(0, 0.12, 0.5));
        assertThat(received.get()).contains(
            "\"type\":\"M\"", "\"airTemperature\":300.0", "\"processTemperature\":310.5",
            "\"rotationalSpeed\":1400", "\"torque\":55.0", "\"toolWear\":180");
    }

    @Test
    void upstreamErrorStatusBecomesMLServiceException() throws Exception {
        String url = startServer(422, "{\"detail\":[]}", 0, new AtomicReference<>());

        assertThatThrownBy(() -> client(url).predict(REQUEST))
            .isInstanceOf(MLServiceException.class)
            .hasMessageContaining("422");
    }

    @Test
    void incompleteResponseIsRejected() throws Exception {
        String url = startServer(200, "{\"prediction\":1}", 0, new AtomicReference<>());

        assertThatThrownBy(() -> client(url).predict(REQUEST)).isInstanceOf(MLServiceException.class);
    }

    @Test
    void slowServiceTimesOut() throws Exception {
        String url = startServer(200,
            "{\"prediction\":0,\"failureProbability\":0.1,\"threshold\":0.5}", 2000, new AtomicReference<>());

        assertThatThrownBy(() -> client(url).predict(REQUEST))
            .isInstanceOf(MLServiceUnavailableException.class);
    }

    @Test
    void unreachableServiceIsReportedAsUnavailable() throws Exception {
        int freePort;
        try (ServerSocket s = new ServerSocket(0)) {
            freePort = s.getLocalPort();
        }

        assertThatThrownBy(() -> client("http://127.0.0.1:" + freePort).predict(REQUEST))
            .isInstanceOf(MLServiceUnavailableException.class);
    }
}
