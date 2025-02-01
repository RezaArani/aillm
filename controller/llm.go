// Copyright (c) 2025 John Doe
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package aillm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/redis/go-redis/v9"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

// Init initializes the LLMContainer by configuring memory management, embedding settings,
// transcriber configurations, and connecting to the Redis database.
//
// This function sets default parameters for temperature, token threshold, RAG configurations,
// fallback responses, and initializes essential components like the memory manager and transcriber.
//
// Returns:
//   - error: An error if Redis configuration is missing or if the Redis connection fails.
func (llm *LLMContainer) Init() error {
	var err error

	// Default Semantic search algorithm
	if llm.SearchAlgorithm == 0 {
		llm.SearchAlgorithm = SimilaritySearch
	}

	// Initialize memory management with a capacity of 300 entries

	llm.MemoryManager = NewMemoryManager(300)
	// Configure text embedding parameters with chunking settings

	ec := EmbeddingConfig{
		ChunkSize:    2048, // Size of each text chunk
		ChunkOverlap: 100,  // Overlap between consecutive chunks for context retention
	}

	llm.EmbeddingConfig = ec

	// Retrieve Tika service URL from environment variables for text processing

	llm.Transcriber.TikaURL = os.Getenv("TikaURL")
	if llm.Transcriber.TikaURL == "" {
		log.Println("Warning: Tika host configuration is missing. As a result, the transcriber will be restricted to processing only text and HTML files.")

	}
	// Initialize the transcriber component

	llm.Transcriber.init()
	// Load Redis configuration from environment variables if not already set

	if llm.RedisClient.Host == "" {
		llm.RedisClient.Host = os.Getenv("REDIS_HOST")
		llm.RedisClient.Password = os.Getenv("REDIS_PASSWORD")
	}
	// Check if Redis host is configured, return an error if missing

	if llm.RedisClient.Host == "" {
		return errors.New("missing redis host configuration")
	}

	// Establish a connection to the Redis server
	llm.RedisClient.redisClient = redis.NewClient(&redis.Options{
		Addr:     llm.RedisClient.Host,
		Password: llm.RedisClient.Password,
		DB:       0,
	})
	ctx := context.Background()
	// Test Redis connection
	_, err = llm.RedisClient.redisClient.Ping(ctx).Result()
	if err != nil {
		return fmt.Errorf("unable to connect to redis host. \n%v", err)
	}
	// predefine basic values
	if llm.Temperature == 0 {
		llm.Temperature = 0.01
	}
	if llm.TopP == 0 {
		llm.TopP = 0.01
	}

	if llm.ScoreThreshold == 0 {
		llm.ScoreThreshold = 0.75
	}

	if llm.RagRowCount == 0 {
		llm.RagRowCount = 5
	}

	if llm.AnswerLanguage == "" {
		llm.AnswerLanguage = "English"
	}

	if llm.NoRagErrorMessage == "" {
		llm.NoRagErrorMessage = "You have to say sadly I don't have any data."
	}

	if llm.NotRelatedAnswer == "" {
		llm.NotRelatedAnswer = "I can't find any answer regarding your question."
	}

	return err
}

// AskLLM processes a user query and retrieves an AI-generated response using Retrieval-Augmented Generation (RAG).
//
// This function supports multi-step processes:
//   - Retrieves session memory to provide context for the query.
//   - Uses a semantic search algorithm (Cosine Similarity or K-Nearest Neighbors) to fetch relevant documents.
//   - Constructs the query for the LLM based on user input and past interactions.
//   - Calls the LLM to generate a response, optionally streaming results to a callback function.
//   - Updates session memory with the query if relevant documents are found.
//
// Parameters:
//   - Query: The user's input query.
//   - options: A variadic parameter to specify additional options like session ID, language, and streaming functions.
//
// Returns:
//   - LLMResult: Struct containing the AI-generated response, retrieved documents, session memory, and logged actions.
//   - error: An error if the query fails or if essential components are missing.

func (llm *LLMContainer) AskLLM(Query string, options ...LLMCallOption) (LLMResult, error) {

	result := LLMResult{}

	// Retrieve memory for the session

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}
	result.addAction("Start Calling LLM", o.ActionCallFunc)

	mem, exists := llm.MemoryManager.GetMemory(o.SessionID)

	ctx := context.Background()
	memoryAddAllowed := false
	llmclient, err := llm.LLMClient.NewLLMClient()
	var msgs []llms.MessageContent
	hasRag := false
	var resDocs interface{}

	// check exact prompt provided or not
	if o.ExactPrompt == "" {
		// Check if LLM client is available

		if llm.LLMClient == nil {
			return result, errors.New("missing llm client")
		}
		// Check if embedding model is available

		if llm.Embedder == nil {
			return result, errors.New("missing embedding model")
		} else {
			// Initialize embedding model if not already initialized

			if !llm.Embedder.initialized() {
				llm.InitEmbedding()
			}
		}
		// Initialize the LLM client for processing
		result.addAction("Vector Search Start", o.ActionCallFunc)

		if err != nil {
			return result, err
		}
		// Add AI assistant's character/personality setting
		if llm.Character != "" {
			msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeSystem, llm.Character))
		}
		// Construct the query prefix for the embedding store
		KNNPrefix := o.getEmbeddingPrefix() + ":"
		if o.searchAll {
			// o.Prefix =
			KNNPrefix = "all:" + o.searchAllLanguage + ":"
		} else {
			if o.Language == "" {
				if llm.FallbackLanguage != "" {
					o.Language = llm.FallbackLanguage
				} else {
					// return result, errors.New("missing language. you can use fallback language if you want to use backup contents")
					o.Language = "en"
				}
			}
			KNNPrefix += o.Language + ":"
		}
		KNNQuery := Query

		// Append past session queries to provide context

		for _, memoryItem := range mem.Questions {
			KNNQuery += "\n" + memoryItem
		}
		// KNNQuery += Query

		/*** Change algorithm to The k-nearest neighbors (KNN) algorithm **/
		var KNNGetErr error

		switch llm.SearchAlgorithm {
		case SimilaritySearch:
			// Retrieve related documents using cosine similarity search

			resDocs, KNNGetErr = llm.CosineSimilarity(KNNPrefix, KNNQuery, llm.RagRowCount, llm.ScoreThreshold)
		case KNearestNeighbors:
			// Retrieve related documents using KNN search
			resDocs, KNNGetErr = llm.FindKNN(KNNPrefix, KNNQuery, llm.RagRowCount, llm.ScoreThreshold)
		default:
			return result, errors.New("unknown search algorithm")
		}

		if KNNGetErr != nil {
			if !llm.AllowHallucinate {
				return result, KNNGetErr
			}
		}
		// Check if relevant documents were retrieved
		hasRag = resDocs != nil && len(resDocs.([]schema.Document)) > 0 && o.ExtraContext == ""

		if !hasRag && llm.FallbackLanguage != "" && llm.FallbackLanguage != o.Language {
			searchPrefix := o.getEmbeddingPrefix()+":"+llm.FallbackLanguage+":"
			if o.searchAll {
				// o.Prefix =
				searchPrefix = "all:" + llm.FallbackLanguage + ":"
			}
			switch llm.SearchAlgorithm {
			case SimilaritySearch:
				resDocs, KNNGetErr = llm.CosineSimilarity(searchPrefix, KNNQuery, llm.RagRowCount, llm.ScoreThreshold)
			case KNearestNeighbors:
				resDocs, KNNGetErr = llm.FindKNN(searchPrefix, KNNQuery, llm.RagRowCount, llm.ScoreThreshold)
			default:
				return result, errors.New("unknown search algorithm")
			}

			// resDocs, KNNGetErr := llm.CosineSimilarity(KNNPrefix, KNNQuery, llm.RagRowCount, llm.ScoreThreshold)
			// resDocs, KNNGetErr = llm.FindKNN(prefix+":"+llm.FallbackLanguage+":", Query, llm.RagRowCount, llm.ScoreThreshold)
			if KNNGetErr != nil {
				if !llm.AllowHallucinate {
					return result, KNNGetErr
				}
			}
			hasRag = resDocs != nil && len(resDocs.([]schema.Document)) > 0

		}
		result.addAction("Prompt Generation Start", o.ActionCallFunc)

		var curMessageContent llms.MessageContent
		var ragArray []llms.ContentPart
		ragText := ""

		// Prepare the language detection function based on the query
		languageCapabilityDetectionFunction := `detect language of "` + Query + `"`
		languageCapabilityDetectionText := `detected language without mentioning it.`
		if llm.LLMModelLanguageDetectionCapability {
			languageCapabilityDetectionFunction = `{language} = detect_language("` + Query + `") without mentionning in response `
			languageCapabilityDetectionText = "{language}"

		} else {
			if llm.AnswerLanguage != "" {
				languageCapabilityDetectionFunction = ""
				languageCapabilityDetectionText = llm.AnswerLanguage
			}
		}

		// If no relevant documents found, handle response accordingly

		if !hasRag && o.ExtraContext == "" {
			if !llm.AllowHallucinate {
				if llm.NoRagErrorMessage != "" {
					ragText = languageCapabilityDetectionFunction + `You are an AI assistant, Think step-by-step before answer.
your only answer to all of questions is the improved version of "` + llm.NotRelatedAnswer + `" in ` + languageCapabilityDetectionText + `.
Assistant:`

					msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeSystem, ragText))
				} else {
					return result, errors.New("rag query has no results and hallucination is allowed but NoRagErrorMessage is empty")
				}
			}
		} else {
			for idx, doc := range resDocs.([]schema.Document) {
				if idx > 0 {
					ragText += "\n"
				}
				ragText += doc.PageContent
			}
			ragText += "\n" + o.ExtraContext
			ragText = languageCapabilityDetectionFunction +
				`\nYou are an AI assistant with knowledge only and only just this text: "` + ragText + `". ` +
				`\nThink step-by-step and then answer briefly in ` + languageCapabilityDetectionText + `. If question is outside this scope, add "@" to the beginning of response and Just answer in ` + languageCapabilityDetectionText + ` something similar to "` + llm.NotRelatedAnswer + ` without mentioning original text or language information."` +
				`\nUser: "` + Query + `"` +
				`\nAssistant:`
			ragArray = append(ragArray, llms.TextPart(ragText))
			curMessageContent.Parts = ragArray
			curMessageContent.Role = llms.ChatMessageTypeSystem
			msgs = append(msgs, curMessageContent)
		}

		// Include session history in the query

		QueryWithMemory := Query

		if len(mem.Questions) > 0 {
			QueryWithMemory += "Here is the context from our previous interactions:\n"
			for idx, q := range mem.Questions {
				QueryWithMemory += "\t" + strconv.Itoa(idx) + "." + q + "\n"
			}

		}

		msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeHuman, QueryWithMemory))
		memoryAddAllowed = hasRag
	} else {
		msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeHuman, o.ExactPrompt))
	}
	isFirstWord := true
	result.addAction("Sending Request to LLM", o.ActionCallFunc)
	isFirstChunk := true
	// Generate content using the LLM and stream results via the provided callback function
	response, err := llmclient.GenerateContent(ctx,
		msgs,
		llms.WithTemperature(llm.Temperature),
		llms.WithTopP(llm.TopP),
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			if isFirstChunk {
				isFirstChunk = false
				result.addAction("First Chunk Received", o.ActionCallFunc)
			}
			if isFirstWord && len(chunk) > 0 {
				startsWithAt := chunk[0] == 64
				if memoryAddAllowed && startsWithAt {
					memoryAddAllowed = false
				}
				isFirstWord = isFirstWord && chunk[0] != 32
				if isFirstWord && startsWithAt {
					chunk = chunk[1:]
				}
			}
			if o.StreamingFunc == nil {
				return nil
			}
			return o.StreamingFunc(ctx, chunk)
		}),
	)
	result.addAction("Finished", o.ActionCallFunc)

	// Update memory with the new query if RAG data was found
	if hasRag && memoryAddAllowed {
		if exists {
			mem.Questions = append(mem.Questions, Query)
			llm.MemoryManager.AddMemory(o.SessionID, mem.Questions)
		} else {
			llm.MemoryManager.AddMemory(o.SessionID, []string{Query})
		}
	}
	result = LLMResult{
		Prompt:   msgs,
		Response: response,
		RagDocs:  resDocs,
		Memory:   mem.Questions[:],
		Actions:  result.Actions,
	}
	return result, err
}

// WithStreamingFunc specifies a callback function for handling streaming output during query processing.
//
// Parameters:
//   - streamingFunc: A function to process streaming chunks of output from the LLM.
//     Returning an error stops the streaming process.
//
// Returns:
//   - LLMCallOption: An option to set the streaming function.
func (llm *LLMContainer) WithStreamingFunc(streamingFunc func(ctx context.Context, chunk []byte) error) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.StreamingFunc = streamingFunc
	}
}

// WithActionCallFunc specifies a callback function to log custom actions during LLM query processing.
//
// Parameters:
//   - ActionCallFunc: A function defining custom actions to be logged during query processing.
//
// Returns:
//   - LLMCallOption: An option to set the custom action callback function.
func (llm *LLMContainer) WithActionCallFunc(ActionCallFunc func(action LLMAction)) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.ActionCallFunc = ActionCallFunc
	}
}

// WithLanguage specifies the language to use for the query response.
//
// Parameters:
//   - Language: The language in which the LLM should generate responses.
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithLanguage(Language string) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.Language = Language
	}
}

// WithSessionID specifies the session ID for tracking user chat history and interactions.
//
// Parameters:
//   - SessionID: A unique identifier for the user's session.
//
// Returns:
//   - LLMCallOption: An option that sets the session ID.
func (llm *LLMContainer) WithSessionID(SessionID string) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.SessionID = SessionID
	}
}

// WithEmbeddingPrefix specifies a prefix for identifying related embeddings.
//
// Parameters:
//   - Prefix: A string prefix used to group or identify embeddings in the store.
//
// Returns:
//   - LLMCallOption: An option that sets the embedding prefix.
func (llm *LLMContainer) WithEmbeddingPrefix(Prefix string) LLMCallOption {
	return func(o *LLMCallOptions) {
		if Prefix == "" {
			Prefix = "default"
		}
		o.Prefix = Prefix
	}
}
func (llm *LLMContainer) SearchAll(language string) LLMCallOption {
	return func(o *LLMCallOptions) {
		// o.Prefix = "all:"+language
		o.searchAllLanguage = language
		o.searchAll = true
	}
}

// WithExtraExtraContext specifies a extra context for search
//
// Parameters:
//   - ExtraContext: Extra provided text to provide LLM.
//
// Returns:
//   - LLMCallOption: An option that sets the embedding prefix.
func (llm *LLMContainer) WithExtraContext(ExtraContext string) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.ExtraContext = ExtraContext
	}
}

// WithExtraExactPromot queries LLM with exact promot.
//
// Parameters:
//   - ExactPromot: string, prompt
//
// Returns:
//   - LLMCallOption: An option that sets the embedding prefix.
func (llm *LLMContainer) WithExactPrompt(ExactPrompt string) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.ExactPrompt = ExactPrompt
	}
}

func (o *LLMCallOptions) getEmbeddingPrefix() string {
	if o.Prefix == "" {
		o.Prefix = "default"
	}
	return "context:" + o.Prefix
}
