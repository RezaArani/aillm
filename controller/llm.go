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
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"

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
	llm.initPersistentMemoryManager()

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
	if o.Prefix == "" {
		o.searchAll = true
	}
	result.addAction("Start Calling LLM", o.ActionCallFunc)
	memoryStr := ""
	KNNMemoryStr := ""
	exists := false
	var memoryData []MemoryData
	var persistentMemoryHistory []schema.Document
	if o.SessionID != "" {

		if !o.PersistentMemory {
			mem, smExists := llm.MemoryManager.GetMemory(o.SessionID)
			for _, memoryItem := range mem.Questions {
				KNNMemoryStr += "\n" + memoryItem.Question
			}
			memoryData = mem.Questions

			exists = smExists
		} else {
			// gget memory data:
			lastQuery := MemoryData{}
			lastQuery, memoryStr,persistentMemoryHistory, _ = llm.PersistentMemoryManager.GetMemory(o.SessionID, Query)
			KNNMemoryStr += lastQuery.Question + " " + lastQuery.Answer
		}
	}
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
		KNNPrefix := "context:" + o.getEmbeddingPrefix() + ":"
		if o.searchAll {
			// o.Prefix =
			KNNPrefix = "all:" + o.getEmbeddingPrefix() + ":" + o.searchAllLanguage + ":"
		} else {
			if o.Language == "" {
				if llm.FallbackLanguage != "" {
					o.Language = llm.FallbackLanguage
				}
			}
			KNNPrefix += o.Index + ":" + o.Language + ":"
		}
		KNNQuery := Query

		// Append past session queries to provide context
		if KNNMemoryStr != "" {
			KNNQuery += "\n" + KNNMemoryStr
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
		hasRag = (resDocs != nil && len(resDocs.([]schema.Document)) > 0) || o.ExtraContext != ""

		if !hasRag && llm.FallbackLanguage != "" && llm.FallbackLanguage != o.Language {
			searchPrefix := o.getEmbeddingPrefix() + ":" + llm.FallbackLanguage + ":"
			if o.searchAll {
				// o.Prefix =
				searchPrefix = "all:" + o.Prefix + ":" + llm.FallbackLanguage + ":"
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
			if llm.userLanguage == nil {
				llm.userLanguage = make(map[string]string)
			}
			if llm.userLanguage[o.SessionID] == "" {
				langResponse, langErr := llmclient.GenerateContent(ctx,
					[]llms.MessageContent{

						llms.TextParts(llms.ChatMessageTypeHuman, `What language is "`+Query+`" in? Say just it in one word without ".".`),
					},
					llms.WithTemperature(0))

				if len(langResponse.Choices) > 0 {
					llm.userLanguage[o.SessionID] = langResponse.Choices[0].Content
					if o.LanguageChannel != nil {
						go func() {
							o.LanguageChannel <- llm.userLanguage[o.SessionID]
						}()

					}
				}
				if langErr != nil || llm.userLanguage[o.SessionID] == "" {
					//unable to detect language
					languageCapabilityDetectionFunction = `{language} = detect_language("` + Query + `") without mentionning in response.`
					languageCapabilityDetectionText = "{language}"
				} else {
					// language detected, will be saved for the session.
					languageCapabilityDetectionFunction = ""
					languageCapabilityDetectionText = llm.userLanguage[o.SessionID]
				}
			} else {
				languageCapabilityDetectionFunction = ""
				languageCapabilityDetectionText = llm.userLanguage[o.SessionID]
				if o.LanguageChannel != nil {
					go func() {
						o.LanguageChannel <- llm.userLanguage[o.SessionID]
					}()

				}

			}

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
			} else {
				// allow hallucinate - reload memory
				if memoryStr != "" {
					memStrPrompt := "Here is the context from our previous interactions:\n" + memoryStr
					ragText = fmt.Sprintf(`You are a %s AI assistant with knowledge:
%s
Think step-by-step and then answer briefly in %s.
without mentioning original text or language information.
User: %s
Assistant:`,
						o.character, memStrPrompt, languageCapabilityDetectionText, Query)
					ragArray = append(ragArray, llms.TextPart(ragText))
					curMessageContent.Parts = ragArray
					curMessageContent.Role = llms.ChatMessageTypeSystem
					msgs = append(msgs, curMessageContent)

				}
			}
		} else {
			for idx, doc := range resDocs.([]schema.Document) {
				if idx > 0 {
					ragText += "\n"
				}
				content := doc.PageContent
				if o.CotextCleanup {
					re := regexp.MustCompile(`<[^>]+>`)
					content = re.ReplaceAllString(content, "")

					// جایگزینی فضاهای تکراری با یک فضای واحد
					reSpaces := regexp.MustCompile(`\s+`)
					content = reSpaces.ReplaceAllString(content, " ")

					// حذف خطوط خالی تکراری
					reNewlines := regexp.MustCompile(`\n+`)
					content = reNewlines.ReplaceAllString(content, "\n")

					// حذف فاصله‌های اضافی ابتدا و انتها
					content = strings.TrimSpace(content)
				}
				ragText += content
			}
			ragText += "\n" + o.ExtraContext
			memStrPrompt := ""
			if memoryStr != "" {
				memStrPrompt = "Here is the context from our previous interactions:\n" + memoryStr
			}
			ragText = fmt.Sprintf(`You are a %s AI assistant with knowledge:
%s

%s

Think step-by-step and then answer briefly in %s.
If question is outside this scope, add "@" to the beginning of response and Just answer in %s something similar to 
"I can't find any answer regarding your question." 
without mentioning original text or language information.

User: %s
Assistant:`,
				o.character, ragText, memStrPrompt, languageCapabilityDetectionText, languageCapabilityDetectionText, Query)

			ragArray = append(ragArray, llms.TextPart(ragText))
			curMessageContent.Parts = ragArray
			curMessageContent.Role = llms.ChatMessageTypeSystem
			msgs = append(msgs, curMessageContent)
		}

		msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeHuman, Query))
		memoryAddAllowed = hasRag || llm.AllowHallucinate
	} else {
		msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeHuman, o.ExactPrompt))
	}
	isFirstWord := true
	isFirstChunk := true
	// Generate content using the LLM and stream results via the provided callback function
	calloptions := []llms.CallOption{
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
	}
	var response *llms.ContentResponse
	if len(o.Tools.Tools) > 0 {
		result.addAction("Calling tools", o.ActionCallFunc)

		messageHistory := []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, memoryStr),
		}
		resp, err := llmclient.GenerateContent(ctx, messageHistory, llms.WithTools(o.Tools.Tools))
		if err != nil {
			return result, err

		}
		respchoice := resp.Choices[0]

		assistantResponse := llms.TextParts(llms.ChatMessageTypeAI, respchoice.Content)
		for _, tc := range respchoice.ToolCalls {
			assistantResponse.Parts = append(assistantResponse.Parts, tc)
		}
		messageHistory = append(messageHistory, assistantResponse)

		for _, tc := range respchoice.ToolCalls {
			if o.Tools.Handlers[tc.FunctionCall.Name] != nil {
				fn := o.Tools.Handlers[tc.FunctionCall.Name]
				var params interface{}
				if err := json.Unmarshal([]byte(tc.FunctionCall.Arguments), &params); err != nil {
					log.Fatal(err)
				}
				fnresult, handlererr := fn(params)
				if handlererr != nil {
					return result, handlererr
				}
				toolResponse := llms.MessageContent{
					Role: llms.ChatMessageTypeTool,
					Parts: []llms.ContentPart{
						llms.ToolCallResponse{
							Name:    tc.FunctionCall.Name,
							Content: fnresult,
						},
					},
				}

				messageHistory = append(messageHistory, toolResponse)
			}
		}

		result.addAction("Sending Request to LLM", o.ActionCallFunc)

		response, err = llmclient.GenerateContent(ctx,
			msgs,
			calloptions...,
		)
		if err != nil {
			return result, err
		}

	} else {
		result.addAction("Sending Request to LLM", o.ActionCallFunc)
		response, err = llmclient.GenerateContent(ctx,
			msgs,
			calloptions...,
		)
	}

	result.addAction("Finished", o.ActionCallFunc)

	memoryAddAllowed = memoryAddAllowed && o.SessionID != ""
	if response != nil {

		// Update memory with the new query if RAG data was found
		if (hasRag || llm.AllowHallucinate) && memoryAddAllowed && response.Choices != nil && len(response.Choices) > 0 {
			queryData := MemoryData{
				Question: Query,
				Answer:   response.Choices[0].Content,
			}

			if !o.PersistentMemory {
				//plain memory
				if exists {
					memoryData = append(memoryData, queryData)
				}
				llm.MemoryManager.AddMemory(o.SessionID, memoryData)

			} else {
				//persistent memory
				llm.PersistentMemoryManager.AddMemory(o.SessionID, queryData)

			}
		}
	}
	if o.PersistentMemory{
		for _,memdoc:= range persistentMemoryHistory{
			// page memdoc.PageContent
			memoryData = append(memoryData, extractMemoryData(memdoc.PageContent))
			// memoryData.Keys = append(memoryData.Keys, memdoc.Metadata["keys"])
			
		}
	}
	result = LLMResult{
		Prompt:   msgs,
		Response: response,
		RagDocs:  resDocs,
		Memory:   memoryData[:],
		Actions:  result.Actions,
	}
	return result, err
}


func extractMemoryData(input string) MemoryData {
	// Variable to store memory data
	var memoryData MemoryData

	// Split the input string based on "Assistant:"
	parts := strings.Split(input, "Assistant:")
	if len(parts) < 2 {
		return memoryData // Return empty if the input string doesn't have the expected structure
	}

	// Extract the part after "User:" and store it in Question
	userPart := strings.TrimSpace(parts[0])
	memoryData.Question = strings.TrimPrefix(userPart, "User:")

	// Extract the part after "Assistant:" and store it in Answer
	assistantPart := strings.TrimSpace(parts[1])
	memoryData.Answer = assistantPart

	// Here, you can add logic to extract Keys or any other data
	// For example, let's assume we extract Keys based on new lines
	memoryData.Keys = strings.Split(assistantPart, "\n")

	return memoryData
}