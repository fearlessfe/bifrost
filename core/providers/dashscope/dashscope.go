// Package dashscope implements the Alibaba Cloud DashScope (Bailian) provider.
//
// Chat and transcription ride DashScope's OpenAI-compatible surface
// (/compatible-mode) via the shared openai handlers. TTS and images speak the
// native DashScope protocol, which compatible-mode does not cover: TTS is a
// multimodal-generation call answering with a download URL (or an SSE stream
// of base64 chunks), and image generation is either a synchronous
// multimodal-generation call (qwen-image*/z-image*) or the legacy asynchronous
// task API with submit + poll (wan*/wanx*).
package dashscope

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// DashScopeProvider implements the Provider interface for Alibaba Cloud DashScope.
type DashScopeProvider struct {
	logger              schemas.Logger        // Logger for provider operations
	client              *fasthttp.Client      // HTTP client for unary API requests (ReadTimeout bounds overall response)
	streamingClient     *fasthttp.Client      // HTTP client for streaming API requests (no ReadTimeout; idle governed by NewIdleTimeoutReader)
	networkConfig       schemas.NetworkConfig // Network configuration including extra headers
	sendBackRawRequest  bool                  // Whether to include raw request in BifrostResponse
	sendBackRawResponse bool                  // Whether to include raw response in BifrostResponse
}

// NewDashScopeProvider creates a new DashScope provider instance.
func NewDashScopeProvider(config *schemas.ProviderConfig, logger schemas.Logger) (*DashScopeProvider, error) {
	config.CheckAndSetDefaults()

	requestTimeout := time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds)
	client := &fasthttp.Client{
		ReadTimeout:         requestTimeout,
		WriteTimeout:        requestTimeout,
		MaxConnsPerHost:     config.NetworkConfig.MaxConnsPerHost,
		MaxIdleConnDuration: time.Second * time.Duration(config.NetworkConfig.KeepAliveTimeoutInSeconds),
		MaxConnWaitTimeout:  requestTimeout,
		MaxConnDuration:     time.Second * time.Duration(schemas.DefaultMaxConnDurationInSeconds),
		ConnPoolStrategy:    fasthttp.FIFO,
	}

	client = providerUtils.ConfigureProxy(client, config.ProxyConfig, logger)
	client = providerUtils.ConfigureDialer(client, config.NetworkConfig.AllowPrivateNetwork)
	client = providerUtils.ConfigureTLS(client, config.NetworkConfig, logger)
	streamingClient := providerUtils.BuildStreamingClient(client)

	if config.NetworkConfig.BaseURL == "" {
		config.NetworkConfig.BaseURL = DefaultDashScopeBaseURL
	}
	config.NetworkConfig.BaseURL = strings.TrimRight(config.NetworkConfig.BaseURL, "/")

	return &DashScopeProvider{
		logger:              logger,
		client:              client,
		streamingClient:     streamingClient,
		networkConfig:       config.NetworkConfig,
		sendBackRawRequest:  config.SendBackRawRequest,
		sendBackRawResponse: config.SendBackRawResponse,
	}, nil
}

// GetProviderKey returns the provider identifier for DashScope.
func (provider *DashScopeProvider) GetProviderKey() schemas.ModelProvider {
	return schemas.DashScope
}

// Speech performs a text-to-speech request to the qwen-tts family via the
// native multimodal-generation endpoint. The upstream answers with a download
// URL (24h validity); the audio is fetched and returned inline.
func (provider *DashScopeProvider) Speech(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostSpeechRequest) (*schemas.BifrostSpeechResponse, *schemas.BifrostError) {
	if request == nil || request.Input == nil || request.Input.Input == "" {
		return nil, providerUtils.NewBifrostOperationError("speech input text is required", nil)
	}

	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			return ToDashScopeTTSRequest(request), nil
		})
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)

	req.SetRequestURI(provider.networkConfig.BaseURL + nativeMultimodalGenerationPath)
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")
	req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	if !providerUtils.ApplyLargePayloadRequestBodyWithModelNormalization(ctx, req, schemas.DashScope) {
		req.SetBody(jsonData)
	}

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, providerUtils.EnrichError(ctx, parseDashScopeError(resp), jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	body, bifrostErr := decodeResponseBody(ctx, resp)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	var ttsResp DashScopeTTSResponse
	if err := sonic.Unmarshal(body, &ttsResp); err != nil {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("failed to parse DashScope text-to-speech response", err), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}
	if ttsResp.Output.Audio.URL == "" {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("DashScope text-to-speech response contained no audio URL", nil), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	// The download is part of serving the request, so its time rides the
	// reported latency alongside the API call.
	downloadStart := time.Now()
	audioBytes, bifrostErr := provider.downloadAudio(ctx, ttsResp.Output.Audio.URL)
	latency += time.Since(downloadStart)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	bifrostResponse := &schemas.BifrostSpeechResponse{
		Audio: audioBytes,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Latency:                 latency.Milliseconds(),
			ProviderResponseHeaders: providerUtils.ExtractProviderResponseHeaders(resp),
		},
	}
	if ttsResp.Usage != nil {
		bifrostResponse.Usage = &schemas.SpeechUsage{
			InputTokens:  ttsResp.Usage.InputTokens,
			OutputTokens: ttsResp.Usage.OutputTokens,
			TotalTokens:  ttsResp.Usage.TotalTokens,
		}
	}

	if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
		providerUtils.ParseAndSetRawRequest(&bifrostResponse.ExtraFields, jsonData)
	}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		var rawResponse interface{}
		if err := sonic.Unmarshal(body, &rawResponse); err != nil {
			rawResponse = string(body)
		}
		bifrostResponse.ExtraFields.RawResponse = rawResponse
	}

	return bifrostResponse, nil
}

// SpeechStream performs a streaming text-to-speech request. With the
// X-DashScope-SSE header the upstream answers with SSE events whose
// output.audio.data carries base64 audio chunks, terminating with a
// finish_reason event that carries usage.
func (provider *DashScopeProvider) SpeechStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostSpeechRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if request == nil || request.Input == nil || request.Input.Input == "" {
		return nil, providerUtils.NewBifrostOperationError("speech input text is required", nil)
	}

	jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			return ToDashScopeTTSRequest(request), nil
		})
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true
	defer fasthttp.ReleaseRequest(req)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)

	req.SetRequestURI(provider.networkConfig.BaseURL + nativeMultimodalGenerationPath)
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")
	req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	req.Header.Set(headerXDashScopeSSE, headerValueEnable)
	if !providerUtils.ApplyLargePayloadRequestBodyWithModelNormalization(ctx, req, schemas.DashScope) {
		req.SetBody(jsonBody)
	}

	startTime := time.Now()
	err := providerUtils.DoStreamingRequest(ctx, provider.streamingClient, req, resp)
	latency := time.Since(startTime)
	if err != nil {
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)
		if errors.Is(err, context.Canceled) {
			return nil, providerUtils.EnrichError(ctx, &schemas.BifrostError{
				IsBifrostError: false,
				Error: &schemas.ErrorField{
					Type:    schemas.Ptr(schemas.RequestCancelled),
					Message: schemas.ErrRequestCancelled,
					Error:   err,
				},
			}, jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
		}
		if errors.Is(err, fasthttp.ErrTimeout) || errors.Is(err, context.DeadlineExceeded) {
			return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostTimeoutError(schemas.ErrProviderRequestTimedOut, err), jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
		}
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostUpstreamConnectionError(schemas.ErrProviderDoRequest, err), jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

	if resp.StatusCode() != fasthttp.StatusOK {
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)
		// Not MaterializeStreamErrorBody: that helper no-ops unless the
		// enterprise large-response middleware set a threshold on the context,
		// which never happens under OSS. An error body is small enough (512KB
		// cap) to always materialize, so the threshold gate is skipped here.
		materializeDashScopeStreamErrorBody(resp)
		return nil, providerUtils.EnrichError(ctx, parseDashScopeError(resp), jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	responseChan := make(chan *schemas.BifrostStreamChunk, schemas.DefaultStreamBufferSize)
	providerUtils.SetStreamIdleTimeoutIfEmpty(ctx, provider.networkConfig.StreamIdleTimeoutInSeconds)

	go func() {
		defer providerUtils.EnsureStreamFinalizerCalled(ctx, postHookSpanFinalizer)
		defer func() {
			if ctx.Err() == context.Canceled {
				providerUtils.HandleStreamCancellation(ctx, postHookRunner, responseChan, provider.logger, postHookSpanFinalizer, jsonBody)
			} else if ctx.Err() == context.DeadlineExceeded {
				providerUtils.HandleStreamTimeout(ctx, postHookRunner, responseChan, provider.logger, postHookSpanFinalizer, jsonBody)
			}
			providerUtils.CloseStream(ctx, responseChan)
		}()
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)

		reader, releaseGzip := providerUtils.DecompressStreamBody(resp)
		defer releaseGzip()

		reader, stopIdleTimeout := providerUtils.NewIdleTimeoutReader(reader, resp.BodyStream(), providerUtils.GetStreamIdleTimeout(ctx), ctx)
		defer stopIdleTimeout()

		stopCancellation := providerUtils.SetupStreamCancellation(ctx, resp.BodyStream(), provider.logger)
		defer stopCancellation()

		// DashScope answers SSE on this surface (not a raw octet stream), so a
		// non-SSE body means the upstream misbehaved.
		reader, nonSSE := providerUtils.DrainNonSSEStreamReader(resp, reader)
		if nonSSE != nil {
			ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
			providerUtils.ProcessAndSendNonSSEStreamError(ctx, postHookRunner, nonSSE, responseChan, provider.logger, postHookSpanFinalizer)
			return
		}

		sseReader := providerUtils.GetSSEDataReader(ctx, reader)
		chunkIndex := -1
		lastChunkTime := time.Now()
		sawTerminalEvent := false

		for {
			if ctx.Err() != nil {
				return
			}

			data, readErr := sseReader.ReadDataLine()
			if readErr != nil {
				if ctx.Err() != nil {
					return
				}
				if readErr != io.EOF {
					ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
					provider.logger.Warn("Error reading DashScope speech stream: %v", readErr)
					providerUtils.ProcessAndSendError(ctx, postHookRunner, readErr, responseChan, provider.logger, postHookSpanFinalizer)
					// The read error chunk is already the terminal signal; falling
					// through to the truncated check would emit a second error.
					return
				}
				break
			}

			var event DashScopeTTSResponse
			if err := sonic.Unmarshal(data, &event); err != nil {
				provider.logger.Warn("Failed to parse DashScope speech stream event: %v", err)
				continue
			}

			if event.Code != "" && event.Output.Audio.Data == "" {
				ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
				streamErr := providerUtils.NewBifrostOperationError(event.Message, nil)
				streamErr.Error.Code = schemas.Ptr(event.Code)
				providerUtils.ProcessAndSendBifrostError(ctx, postHookRunner, streamErr, responseChan, provider.logger, postHookSpanFinalizer)
				return
			}

			if event.Output.Audio.Data != "" {
				audioChunk, err := base64.StdEncoding.DecodeString(event.Output.Audio.Data)
				if err != nil {
					ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
					providerUtils.ProcessAndSendError(ctx, postHookRunner, fmt.Errorf("failed to decode DashScope speech audio chunk: %w", err), responseChan, provider.logger, postHookSpanFinalizer)
					return
				}

				chunkIndex++
				response := &schemas.BifrostSpeechStreamResponse{
					Type:  schemas.SpeechStreamResponseTypeDelta,
					Audio: audioChunk,
					ExtraFields: schemas.BifrostResponseExtraFields{
						ChunkIndex: chunkIndex,
						Latency:    time.Since(lastChunkTime).Milliseconds(),
					},
				}
				lastChunkTime = time.Now()

				if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
					response.ExtraFields.RawResponse = event.Output.Audio.Data
				}

				providerUtils.ProcessAndSendResponse(ctx, postHookRunner, providerUtils.GetBifrostResponseForStreamResponse(nil, nil, nil, response, nil, nil), responseChan, postHookSpanFinalizer)
			}

			// DashScope sends finish_reason as the STRING "null" on non-terminal
			// events (not JSON null), so a plain nil check would end the stream
			// after the first audio chunk. Only a real status (e.g. "stop")
			// terminates.
			if event.Output.FinishReason == nil || *event.Output.FinishReason == "" || *event.Output.FinishReason == "null" {
				continue
			}

			// Terminal event: usage rides it; emit the done chunk and stop.
			sawTerminalEvent = true
			finalResponse := &schemas.BifrostSpeechStreamResponse{
				Type:  schemas.SpeechStreamResponseTypeDone,
				Audio: []byte{},
				ExtraFields: schemas.BifrostResponseExtraFields{
					ChunkIndex: chunkIndex + 1,
					Latency:    time.Since(startTime).Milliseconds(),
				},
			}
			if event.Usage != nil {
				finalResponse.Usage = &schemas.SpeechUsage{
					InputTokens:  event.Usage.InputTokens,
					OutputTokens: event.Usage.OutputTokens,
					TotalTokens:  event.Usage.TotalTokens,
				}
			}
			if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
				providerUtils.ParseAndSetRawRequest(&finalResponse.ExtraFields, jsonBody)
			}
			finalResponse.BackfillParams(request)
			ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
			providerUtils.ProcessAndSendResponse(ctx, postHookRunner, providerUtils.GetBifrostResponseForStreamResponse(nil, nil, nil, finalResponse, nil, nil), responseChan, postHookSpanFinalizer)
			return
		}

		// Falling out of the loop means the body ended before the
		// finish_reason event — a truncated audio clip the caller cannot tell
		// from a healthy close.
		if !sawTerminalEvent {
			providerUtils.SendStreamTruncatedError(ctx, postHookRunner, responseChan, provider.logger, postHookSpanFinalizer, jsonBody)
		}
	}()

	return responseChan, nil
}

// ImageGeneration performs an image generation request. Models are dispatched
// by family: wan*/wanx* run on the legacy asynchronous task API (submit +
// poll), qwen-image*/z-image* and everything else on the synchronous
// multimodal-generation endpoint.
func (provider *DashScopeProvider) ImageGeneration(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageGenerationRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	if request == nil || request.Input == nil || request.Input.Prompt == "" {
		return nil, providerUtils.NewBifrostOperationError("image generation prompt is required", nil)
	}
	if isAsyncImageModel(request.Model) {
		return provider.submitAndPollImageTask(ctx, key, request)
	}
	return provider.generateImageSync(ctx, key, request)
}

// generateImageSync runs the synchronous image path on multimodal-generation.
func (provider *DashScopeProvider) generateImageSync(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageGenerationRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			return ToDashScopeSyncImageRequest(request)
		})
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)

	req.SetRequestURI(provider.networkConfig.BaseURL + nativeMultimodalGenerationPath)
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")
	req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	if !providerUtils.ApplyLargePayloadRequestBodyWithModelNormalization(ctx, req, schemas.DashScope) {
		req.SetBody(jsonData)
	}

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, providerUtils.EnrichError(ctx, parseDashScopeError(resp), jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	body, bifrostErr := decodeResponseBody(ctx, resp)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	var mmResp DashScopeMultimodalResponse
	if err := sonic.Unmarshal(body, &mmResp); err != nil {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("failed to parse DashScope image generation response", err), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	bifrostResponse := ToBifrostSyncImageResponse(&mmResp, request.Model)
	if len(bifrostResponse.Data) == 0 {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("DashScope image generation response contained no images", nil), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}
	bifrostResponse.ExtraFields = schemas.BifrostResponseExtraFields{
		Latency:                 latency.Milliseconds(),
		ProviderResponseHeaders: providerUtils.ExtractProviderResponseHeaders(resp),
	}

	if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
		providerUtils.ParseAndSetRawRequest(&bifrostResponse.ExtraFields, jsonData)
	}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		var rawResponse interface{}
		if err := sonic.Unmarshal(body, &rawResponse); err != nil {
			rawResponse = string(body)
		}
		bifrostResponse.ExtraFields.RawResponse = rawResponse
	}

	return bifrostResponse, nil
}

// submitAndPollImageTask submits a task to the legacy asynchronous image API
// and polls it to a terminal state, mapping the async upstream onto Bifrost's
// synchronous ImageGeneration interface.
func (provider *DashScopeProvider) submitAndPollImageTask(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageGenerationRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	startTime := time.Now()

	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			return ToDashScopeTaskImageRequest(request)
		})
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)

	submitPath := nativeTextToImagePath
	if isImageToImageModel(request.Model) {
		submitPath = nativeImageToImagePath
	}
	req.SetRequestURI(provider.networkConfig.BaseURL + submitPath)
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")
	req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	req.Header.Set(headerXDashScopeAsync, headerValueEnable)
	if !providerUtils.ApplyLargePayloadRequestBodyWithModelNormalization(ctx, req, schemas.DashScope) {
		req.SetBody(jsonData)
	}

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, providerUtils.EnrichError(ctx, parseDashScopeError(resp), jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	body, bifrostErr := decodeResponseBody(ctx, resp)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	var submitResp DashScopeTaskSubmitResponse
	if err := sonic.Unmarshal(body, &submitResp); err != nil {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("failed to parse DashScope task submit response", err), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}
	if submitResp.Output.TaskID == "" {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("DashScope task submit response contained no task id", nil), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	taskResp, rawResponse, bifrostErr := provider.pollImageTask(ctx, key, submitResp.Output.TaskID, providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse))
	// The synchronous facade covers submit + the full poll loop, so the
	// reported latency spans the whole operation, not just the submit call.
	totalLatency := time.Since(startTime)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, totalLatency)
	}

	bifrostResponse := ToBifrostTaskImageResponse(taskResp, request.Model)
	if len(bifrostResponse.Data) == 0 {
		// Surface the upstream failure (e.g. content moderation) when every
		// result was dropped; fall back to the generic message otherwise.
		noImagesErr := FirstTaskImageFailureError(taskResp)
		if noImagesErr == nil {
			noImagesErr = providerUtils.NewBifrostOperationError("DashScope image task completed without images", nil)
		}
		return nil, providerUtils.EnrichError(ctx, noImagesErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, totalLatency)
	}
	bifrostResponse.ExtraFields = schemas.BifrostResponseExtraFields{
		Latency:                 totalLatency.Milliseconds(),
		ProviderResponseHeaders: providerUtils.ExtractProviderResponseHeaders(resp),
	}

	if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
		providerUtils.ParseAndSetRawRequest(&bifrostResponse.ExtraFields, jsonData)
	}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) && rawResponse != nil {
		bifrostResponse.ExtraFields.RawResponse = rawResponse
	}

	return bifrostResponse, nil
}

// dashScopeTaskPollMaxRetries bounds consecutive transient poll failures
// (transport errors, upstream 429/5xx) before the poll gives up.
const dashScopeTaskPollMaxRetries = 2

// dashScopeTaskPollRetryBackoff is the wait between transient poll retries.
const dashScopeTaskPollRetryBackoff = time.Second

// pollImageTask polls a DashScope task until it reaches a terminal state or
// the request timeout elapses.
func (provider *DashScopeProvider) pollImageTask(ctx *schemas.BifrostContext, key schemas.Key, taskID string, sendBackRawResponse bool) (*DashScopeTaskResponse, interface{}, *schemas.BifrostError) {
	// The poll budget reuses DefaultRequestTimeoutInSeconds, so this deadline
	// caps the whole async task's lifetime, not a single poll call: lowering
	// the request timeout also shrinks how long a wan*/wanx* generation may run.
	pollCtx, cancel := schemas.NewBifrostContextWithTimeout(ctx, time.Duration(provider.networkConfig.DefaultRequestTimeoutInSeconds)*time.Second)
	defer cancel()

	ticker := time.NewTicker(dashScopeTaskPollingInterval)
	defer ticker.Stop()

	transientFailures := 0
	for {
		taskResp, rawResponse, bifrostErr := provider.retrieveImageTask(pollCtx, key, taskID, sendBackRawResponse)
		if bifrostErr != nil {
			// A cancelled or expired context ends the poll immediately — the
			// error already reports it, and retrying would only burn budget.
			if pollCtx.Err() != nil {
				return nil, nil, bifrostErr
			}
			// Transient transport/upstream failures get a bounded number of
			// retries with a short backoff; anything else (auth failure, bad
			// request, decode error) is final.
			if transientFailures < dashScopeTaskPollMaxRetries && isTransientPollError(bifrostErr) {
				transientFailures++
				provider.logger.Warn("dashscope task %s poll failed (consecutive failure %d/%d), retrying: %s", taskID, transientFailures, dashScopeTaskPollMaxRetries, bifrostErr)
				select {
				case <-pollCtx.Done():
					return nil, nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderRequestTimedOut, fmt.Errorf("dashscope task polling timed out"))
				case <-time.After(dashScopeTaskPollRetryBackoff):
				}
				continue
			}
			return nil, nil, bifrostErr
		}
		transientFailures = 0

		switch status := DashScopeTaskStatus(taskResp.Output.TaskStatus); status {
		case TaskStatusSucceeded:
			return taskResp, rawResponse, nil
		case TaskStatusFailed, TaskStatusCanceled, TaskStatusUnknown:
			// Terminal task states are never retried: the task itself failed,
			// so polling again returns the same verdict.
			return nil, nil, taskTerminalError(status, taskResp.Output.Code, taskResp.Output.Message)
		case TaskStatusPending, TaskStatusRunning:
			// Non-terminal: keep polling below.
		default:
			// An undocumented status would otherwise burn the whole poll
			// budget before timing out; fail fast and surface it instead.
			return nil, nil, providerUtils.NewBifrostOperationError(fmt.Sprintf("dashscope task returned unrecognized task_status %q", taskResp.Output.TaskStatus), nil)
		}

		select {
		case <-pollCtx.Done():
			return nil, nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderRequestTimedOut, fmt.Errorf("dashscope task polling timed out"))
		case <-ticker.C:
		}
	}
}

// isTransientPollError reports whether a failed poll attempt is worth
// retrying: transport-level failures (timeout, connection reset/refused) and
// upstream 429/5xx responses. Everything else — 4xx refusals, decode
// failures, caller cancellation — is final.
func isTransientPollError(bifrostErr *schemas.BifrostError) bool {
	if bifrostErr == nil {
		return false
	}
	if bifrostErr.Error != nil && bifrostErr.Error.Type != nil {
		switch *bifrostErr.Error.Type {
		case schemas.RequestTimedOut, schemas.ProviderConnectionFailed:
			return true
		case schemas.RequestCancelled:
			return false
		}
	}
	if bifrostErr.StatusCode != nil {
		code := *bifrostErr.StatusCode
		return code == fasthttp.StatusTooManyRequests || code >= fasthttp.StatusInternalServerError
	}
	return false
}

// retrieveImageTask fetches the current state of a DashScope image task.
func (provider *DashScopeProvider) retrieveImageTask(ctx *schemas.BifrostContext, key schemas.Key, taskID string, sendBackRawResponse bool) (*DashScopeTaskResponse, interface{}, *schemas.BifrostError) {
	escapedTaskID, idErr := providerUtils.EscapeResourceID(taskID, "task_id")
	if idErr != nil {
		return nil, nil, idErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)

	req.SetRequestURI(provider.networkConfig.BaseURL + nativeTasksPath + escapedTaskID)
	req.Header.SetMethod(http.MethodGet)
	req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, nil, bifrostErr
	}

	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, nil, providerUtils.SetErrorLatency(parseDashScopeError(resp), latency)
	}

	body, err := providerUtils.CheckAndDecodeBody(resp)
	if err != nil {
		return nil, nil, providerUtils.SetErrorLatency(providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, err), latency)
	}

	var taskResp DashScopeTaskResponse
	_, rawResponse, bifrostErr := providerUtils.HandleProviderResponseCtx(ctx, body, &taskResp, nil, false, sendBackRawResponse)
	if bifrostErr != nil {
		return nil, nil, providerUtils.SetErrorLatency(bifrostErr, latency)
	}

	return &taskResp, rawResponse, nil
}

// ImageEdit performs an image edit request via the qwen-image-edit family on
// the synchronous multimodal-generation endpoint.
func (provider *DashScopeProvider) ImageEdit(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageEditRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	if request == nil || request.Input == nil || len(request.Input.Images) == 0 {
		return nil, providerUtils.NewBifrostOperationError("image edit requires at least one input image", nil)
	}
	if isAsyncImageModel(request.Model) {
		return nil, providerUtils.NewBifrostOperationError("image edit is only supported for the synchronous qwen-image-edit models", nil)
	}

	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			return ToDashScopeImageEditRequest(request), nil
		})
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)

	req.SetRequestURI(provider.networkConfig.BaseURL + nativeMultimodalGenerationPath)
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")
	req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	if !providerUtils.ApplyLargePayloadRequestBodyWithModelNormalization(ctx, req, schemas.DashScope) {
		req.SetBody(jsonData)
	}

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, providerUtils.EnrichError(ctx, parseDashScopeError(resp), jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	body, bifrostErr := decodeResponseBody(ctx, resp)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	var mmResp DashScopeMultimodalResponse
	if err := sonic.Unmarshal(body, &mmResp); err != nil {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("failed to parse DashScope image edit response", err), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	bifrostResponse := ToBifrostSyncImageResponse(&mmResp, request.Model)
	if len(bifrostResponse.Data) == 0 {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError("DashScope image edit response contained no images", nil), jsonData, body, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}
	bifrostResponse.ExtraFields = schemas.BifrostResponseExtraFields{
		Latency:                 latency.Milliseconds(),
		ProviderResponseHeaders: providerUtils.ExtractProviderResponseHeaders(resp),
	}

	if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
		providerUtils.ParseAndSetRawRequest(&bifrostResponse.ExtraFields, jsonData)
	}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		var rawResponse interface{}
		if err := sonic.Unmarshal(body, &rawResponse); err != nil {
			rawResponse = string(body)
		}
		bifrostResponse.ExtraFields.RawResponse = rawResponse
	}

	return bifrostResponse, nil
}

const (
	// dashScopeAudioDownloadMaxBytes bounds the synthesized audio fetch: the
	// download URL is upstream-controlled, so the read must not be unbounded.
	dashScopeAudioDownloadMaxBytes = 100 * 1024 * 1024
	// dashScopeAudioDownloadMaxRedirects caps redirect hops on the download
	// URL (signed OSS links may bounce before serving the bytes).
	dashScopeAudioDownloadMaxRedirects = 5
)

// downloadAudio fetches the synthesized audio from the upstream's download
// URL (an OSS link that needs no auth) so the response can carry the bytes.
func (provider *DashScopeProvider) downloadAudio(ctx *schemas.BifrostContext, audioURL string) ([]byte, *schemas.BifrostError) {
	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	// Stream the body so the size cap is enforced while reading instead of
	// after fasthttp has already buffered the whole download in memory.
	resp.StreamBody = true
	defer fasthttp.ReleaseRequest(req)
	defer providerUtils.ReleaseStreamingResponse(ctx, resp)

	req.SetRequestURI(audioURL)
	req.Header.SetMethod(http.MethodGet)

	_, bifrostErr, wait := providerUtils.MakeRequestWithContextFollowRedirects(ctx, provider.client, req, resp, dashScopeAudioDownloadMaxRedirects)
	defer wait()
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, providerUtils.NewBifrostOperationError(fmt.Sprintf("failed to download synthesized audio (HTTP %d)", resp.StatusCode()), nil)
	}

	// Cheap early-out when the upstream declares an oversized identity body;
	// for gzip the header describes the compressed length, and for chunked
	// there is no header, so the LimitReader below is the real cap.
	if contentLength := resp.Header.ContentLength(); int64(contentLength) > dashScopeAudioDownloadMaxBytes && len(resp.Header.Peek("Content-Encoding")) == 0 {
		abandonDashScopeDownload(ctx, resp)
		return nil, providerUtils.NewBifrostOperationError(fmt.Sprintf("synthesized audio exceeds the %d byte download limit (content-length %d)", dashScopeAudioDownloadMaxBytes, contentLength), nil)
	}

	reader, releaseGzip := providerUtils.DecompressStreamBody(resp)
	defer releaseGzip()

	body, err := io.ReadAll(io.LimitReader(reader, dashScopeAudioDownloadMaxBytes+1))
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, err)
	}
	if int64(len(body)) > dashScopeAudioDownloadMaxBytes {
		abandonDashScopeDownload(ctx, resp)
		return nil, providerUtils.NewBifrostOperationError(fmt.Sprintf("synthesized audio exceeds the %d byte download limit", dashScopeAudioDownloadMaxBytes), nil)
	}
	return body, nil
}

// dashScopeErrAudioDownloadAborted marks an oversized download's connection
// for closure instead of pool reuse.
var dashScopeErrAudioDownloadAborted = errors.New("dashscope audio download aborted: size limit exceeded")

// abandonDashScopeDownload closes an oversized download's body stream with a
// non-nil error, taking fasthttp's CloseConn path so the unread remainder is
// never drained (it is unbounded) and the half-read connection never returns
// to the idle pool. The context flag keeps the deferred
// ReleaseStreamingResponse from double-closing; the response is left to GC,
// the documented trade-off for abandoned streams.
func abandonDashScopeDownload(ctx *schemas.BifrostContext, resp *fasthttp.Response) {
	if prev, _ := ctx.GetAndSetValue(schemas.BifrostContextKeyConnectionClosed, true).(bool); prev {
		return
	}
	if bodyStream := resp.BodyStream(); bodyStream != nil {
		if closer, ok := bodyStream.(interface{ CloseWithError(error) error }); ok {
			_ = closer.CloseWithError(dashScopeErrAudioDownloadAborted)
		}
	}
}

// materializeDashScopeStreamErrorBody reads a streamed error body back into
// resp so parseDashScopeError can inspect it. Same mechanism as
// providerUtils.MaterializeStreamErrorBody (decompress-aware read, 512KB cap,
// SetBody) minus its large-response threshold gate: that threshold only
// exists when the enterprise middleware sets it, and an error body is small
// enough to always materialize, including under OSS.
func materializeDashScopeStreamErrorBody(resp *fasthttp.Response) {
	if resp.BodyStream() == nil {
		return
	}
	reader, releaseGzip := providerUtils.DecompressStreamBody(resp)
	defer releaseGzip()
	bodyBytes, err := io.ReadAll(io.LimitReader(reader, 512*1024))
	if err != nil {
		return
	}
	resp.SetBody(bodyBytes)
}

// decodeResponseBody decodes (and decompresses) a successful response body,
// timing the work as the response-finalize phase.
//
// The unary Speech/Image paths that call this are not wired into the
// large-response streaming machinery (FinalizeResponseWithLargeDetection):
// under that mode they degrade to buffering the full response in memory here.
func decodeResponseBody(ctx *schemas.BifrostContext, resp *fasthttp.Response) ([]byte, *schemas.BifrostError) {
	ft, fh := providerUtils.StartPhaseSpan(ctx, "response-finalize")
	body, err := providerUtils.CheckAndDecodeBody(resp)
	if ft != nil {
		if err != nil {
			ft.EndSpan(fh, schemas.SpanStatusError, err.Error())
		} else {
			ft.EndSpan(fh, schemas.SpanStatusOk, "")
		}
	}
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, err)
	}
	return body, nil
}
