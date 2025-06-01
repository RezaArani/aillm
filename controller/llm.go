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
	"strconv"
	"strings"
	"time"

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

	if llm.EmbeddingConfig.ChunkSize == 0 {
		ec := EmbeddingConfig{
			ChunkSize:    2048, // Size of each text chunk
			ChunkOverlap: 100,  // Overlap between consecutive chunks for context retention
		}
		llm.EmbeddingConfig = ec
	}

	// Retrieve Tika service URL from environment variables for text processing

	llm.Transcriber.TikaURL = os.Getenv("TikaURL")
	if llm.Transcriber.TikaURL == "" && llm.ShowWarnings {
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
		Addr:        llm.RedisClient.Host,
		Password:    llm.RedisClient.Password,
		DB:          0,
		DialTimeout: 5 * time.Second,
	})
	ctx := context.TODO()
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

// GetQueryLanguage Returns user query Language.
//
//
// Parameters:
//   - Query: The user's input query.
//
// Returns:
//   - string: detected Language by LLM Model.
//   - error: An error if the query fails or if essential components are missing.

func (llm *LLMContainer) GetQueryLanguage(Query, sessionId string, languageChannel chan<- string) (string, TokenUsage, error) {
	llmclient, err := llm.LLMClient.NewLLMClient()
	tokenReport := TokenUsage{}
	if err != nil {
		return "", tokenReport, err
	}

	langResponse, langErr := llmclient.GenerateContent(context.TODO(),
		[]llms.MessageContent{

			llms.TextParts(llms.ChatMessageTypeHuman, `What language is "`+Query+`" in? Say just it in one word without "." and just return "NONE" if you can't detect it.`),
		},
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			tokenReport.OutputTokens++
			return nil
		}),
		llms.WithTemperature(0))
	if langErr != nil {
		return "", tokenReport, langErr
	}
	language := langResponse.Choices[0].Content
	switch strings.ToLower(language) {
	case "none":
		language = "English"
	case "portuguese":
		language = "European Portuguese (pt-PT)"
	case "pt":
		language = "European Portuguese (pt-PT)"
	}
	return language, tokenReport, nil

}
func (llm *LLMContainer) setupResponseLanguage(Query, SessionId string, languageChannel chan<- string) (languageCapabilityDetectionFunction, languageCapabilityDetectionText string, LanguageDetectionTokens TokenUsage) {
	if llm.userLanguage == nil {
		llm.userLanguage = make(map[string]string)
	}
	if llm.userLanguage[SessionId] == "" {

		userQueryLanguage, queryLanguageDetectionTokens, detectionError := llm.GetQueryLanguage(Query, SessionId, languageChannel)
		LanguageDetectionTokens = queryLanguageDetectionTokens
		if detectionError == nil && userQueryLanguage != "NONE" {
			llm.userLanguage[SessionId] = userQueryLanguage
		}
		if detectionError != nil || llm.userLanguage[SessionId] == "" {
			//unable to detect language
			languageCapabilityDetectionFunction = `{language} = detect_language("` + Query + `") without mentionning in response.`
			languageCapabilityDetectionText = "{language}"
		} else {
			// language detected, will be saved for the session.
			languageCapabilityDetectionFunction = ""
			languageCapabilityDetectionText = llm.userLanguage[SessionId]
		}
	} else {
		languageCapabilityDetectionFunction = ""
		languageCapabilityDetectionText = llm.userLanguage[SessionId]

	}
	if languageChannel != nil && SessionId != "" {
		go func() {
			defer func() {
				if r := recover(); r != nil {
					// ok = false
					log.Printf("sending language to closed channel, panic recovered: %v\n", r)
				}
			}()
			languageChannel <- llm.userLanguage[SessionId]
		}()

	}
	return languageCapabilityDetectionFunction, languageCapabilityDetectionText, LanguageDetectionTokens

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
	totalTokens := 0
	// Retrieve memory for the session

	o := LLMCallOptions{}
	for _, opt := range options {
		opt(&o)
	}
	if o.Index == "" {
		o.searchAll = true
	}
	result.addAction("Start Calling LLM", o.ActionCallFunc)
	memoryStr := ""
	KNNMemoryStr := ""
	MemorySummary := ""
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
			usermemory := Memory{}
			lastQuery, usermemory, memoryStr, persistentMemoryHistory, _ = llm.PersistentMemoryManager.GetMemory(o.SessionID, Query)
			MemorySummary = usermemory.Summary
			KNNMemoryStr += lastQuery.Question
		}
	}
	ctx := context.Background()
	memoryAddAllowed := false
	llmclient, err := llm.LLMClient.NewLLMClient()
	var msgs []llms.MessageContent
	hasRag := false
	var resDocs []schema.Document
	// Set Date and Time
	datePrompt := ""
	if o.IncludeDate {

		datePrompt = "- It is " + time.Now().Format("Monday, 2006-01-02 15:04") + ". Adjust your response based on the current date and time.\n"
	}
	ragReferencesPrompt := ""
	if o.RagReferences {

		ragReferencesPrompt = `### Output Formatting Rules:
- First, output the **full natural language answer**, formatted clearly.  
- Then, on a **new line after the full answer**, add the **reference line** that begins with **⧉**, followed by a single valid JSON object in this format:  
  ⧉ {"references":["chunk_id_1","chunk_id_2"]}

- The **⧉ line must come immediately after the answer**, with no additional explanation or text.  
- If no references are applicable, **omit the ⧉ line completely** — do not include an empty or placeholder reference object.

- The ⧉ line is used for post-processing and will not be shown to the user. Format it precisely and cleanly.

`
	}
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
		KNNPrefix := "context:"
		if o.getEmbeddingPrefix() != "" {
			KNNPrefix += o.getEmbeddingPrefix() + ":"
		}
		if o.Index == "" {
			o.searchAll = true
		}
		if o.searchAll {
			// o.Prefix =
			KNNPrefix = "all:"
			if o.getEmbeddingPrefix() != "" {
				KNNPrefix += o.getEmbeddingPrefix() + ":"
			}

		} else {
			KNNPrefix += o.Index + ":"
			if o.Language == "" {
				if llm.FallbackLanguage != "" {
					o.Language = llm.FallbackLanguage
				}
			}

		}
		// Issue with forced language. Interference with vector search index!!!! Will be fixed in the future.
		if o.Language != "" && !o.ForceLanguage {
			KNNPrefix += o.Language + ":"
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
			if !llm.AllowHallucinate && !o.AllowHallucinate {
				return result, KNNGetErr
			}
		}
		// Check if relevant documents were retrieved
		hasRag = len(resDocs) > 0

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

			if KNNGetErr != nil {
				if !llm.AllowHallucinate && !o.AllowHallucinate {
					return result, KNNGetErr
				}
			}
		}
		result.addAction("Prompt Generation Start", o.ActionCallFunc)
		hasRag = len(resDocs) > 0 || o.ExtraContext != ""

		var curMessageContent llms.MessageContent
		var ragArray []llms.ContentPart
		ragText := ""

		// Prepare the language detection function based on the query
		languageCapabilityDetectionFunction := ``
		languageCapabilityDetectionText := ``

		if o.ForceLanguage && o.Language != "" {
			languageCapabilityDetectionText = o.Language
		} else {
			languageCapabilityDetectionFunction = `detect language of "` + Query + `"`
			languageCapabilityDetectionText = `detected language without mentioning it.`

			if llm.LLMModelLanguageDetectionCapability {
				LanguageDetectionTokens := TokenUsage{}
				languageCapabilityDetectionFunction, languageCapabilityDetectionText, LanguageDetectionTokens = llm.setupResponseLanguage(Query, o.SessionID, o.LanguageChannel)
				result.TokenReport.LanguageDetectionTokens = LanguageDetectionTokens
			} else {
				if llm.AnswerLanguage != "" {
					languageCapabilityDetectionText = llm.AnswerLanguage
				}
			}
		}

		brieflyText := "briefly "
		if o.ForceLLMToAnswerLong {
			brieflyText = ""
		}
		// If no relevant documents found, handle response accordingly

		if !hasRag && o.ExtraContext == "" {
			if !llm.AllowHallucinate && !o.AllowHallucinate {
				if llm.NoRagErrorMessage != "" {
					ragText = languageCapabilityDetectionFunction + `You are an AI assistant specialized in providing accurate and concise answers.
your only answer to all of questions is the improved version of "` + llm.NotRelatedAnswer + `" in ` + languageCapabilityDetectionText + `.
- Start the response with "@".
- Ignore all of the references and do not include them in the response.
**Assistant:** `

					msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeSystem, ragText))
				} else {
					return result, errors.New("rag query has no results and hallucination is allowed but NoRagErrorMessage is empty")
				}
			} else {
				// allow hallucinate - reload memory
				if memoryStr != "" {
					if MemorySummary != "" {
						memoryStr = MemorySummary + "\n" + memoryStr
					}

					memStrPrompt := `**Previous Interactions:**  
` + memoryStr
					ragText = fmt.Sprintf(`You are a %s AI assistant specialized in providing accurate and concise answers based on the following knowledge:
**Contextual Knowledge:**
%s
**Instructions:** 
- Analyze the question carefully and reason step-by-step.
- Then, provide a **clear answer `+brieflyText+`in %s.**.
- If the question is unrelated to the provided context or cannot be answered based on the information above, **start the response with "@"** and reply politely in %s with something like:  
**"I can't find any answer regarding your question."**. Do not forget to add **@** at the start of the response in case of unanswerable question.
- Do **not** reference the original text or mention language/translation details.
%s
**User:** %s
**Assistant:** `,
						o.character, memStrPrompt, languageCapabilityDetectionText, languageCapabilityDetectionText, datePrompt, Query)
					ragArray = append(ragArray, llms.TextPart(ragText))
					curMessageContent.Parts = ragArray
					curMessageContent.Role = llms.ChatMessageTypeSystem
					msgs = append(msgs, curMessageContent)

				}
			}
		} else {
			for idx, doc := range resDocs {
				if idx > 0 {
					ragText += "\n"
				}
				content := "Chunk " + strconv.Itoa(idx+1) + ":\n"

				if o.RagReferences {
					rawKey := doc.Metadata["rawkey"]

					if rawKey != nil {
						rawKeyObject := LLMEmbeddingContent{}
						err := json.Unmarshal([]byte(rawKey.(string)), &rawKeyObject)
						if err == nil {
							content += `####Reference: ` + rawKeyObject.Id + "\n"
						}
					}
				}
				content += doc.PageContent + "\n\n"
				if o.CotextCleanup {
					re := regexp.MustCompile(`<[^>]+>`)
					content = re.ReplaceAllString(content, "")

					// Replacing repeated spaces with a single space
					reSpaces := regexp.MustCompile(`\s+`)
					content = reSpaces.ReplaceAllString(content, " ")

					// Removing empty lines
					reNewlines := regexp.MustCompile(`\n+`)
					content = reNewlines.ReplaceAllString(content, "\n")

					// Removing extra spaces at the beginning and end
					content = strings.TrimSpace(content)
				}
				ragText += content
			}
			ragText += "\n" + o.ExtraContext
			memStrPrompt := ""
			if memoryStr != "" {
				memStrPrompt = `**Previous Interactions:**  
` + memoryStr
			}
			ragText = fmt.Sprintf(`You are a %s AI assistant specialized in providing accurate and concise answers based on the following knowledge:
**Contextual Knowledge:**			
%s

%s

**Instructions:**
- Analyze the question carefully and reason step-by-step and think about the question and answer first.
- Then, provide a **clear answer `+brieflyText+` in %s.**.
- If the question is unrelated to the provided context or cannot be answered based on the information above, **start the response with "@"** and reply politely in %s with something like:  
**"I can't find any answer regarding your question."**. Do not forget to add **@** at the start of the response in case of unanswerable question.
- Do **not** reference the original text or mention language/translation details.
- Ignore chunk completely if it is not related to the question.
- Do not include chunk number in the response.

%s
%s

**User:** %s
**Assistant:** `,
				o.character, ragText, memStrPrompt, languageCapabilityDetectionText, languageCapabilityDetectionText, datePrompt, ragReferencesPrompt, Query)
			ragArray = append(ragArray, llms.TextPart(ragText))
			curMessageContent.Parts = ragArray
			curMessageContent.Role = llms.ChatMessageTypeSystem
			msgs = append(msgs, curMessageContent)

		}

		msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeHuman, Query))
		memoryAddAllowed = hasRag || llm.AllowHallucinate
	} else {
		if o.ForceLanguage {
			_, Language, _ := llm.setupResponseLanguage(Query, o.SessionID, o.LanguageChannel)
			if Language != "" {
				msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeSystem, "Reply in "+Language))
			}

		}

		msgs = append(msgs, llms.TextParts(llms.ChatMessageTypeHuman, o.ExactPrompt))
	}
	isFirstWord := true
	isFirstChunk := true
	// Generate content using the LLM and stream results via the provided callback function
	refrencesStr := ""
	startRefrences := false
	failedToRespond := false
	calloptions := []llms.CallOption{
		llms.WithTemperature(llm.Temperature),
		llms.WithTopP(llm.TopP),
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			totalTokens++
			if isFirstChunk {
				isFirstChunk = false
				result.addAction("First Chunk Received", o.ActionCallFunc)
			}
			if isFirstWord && len(chunk) > 0 {
				startsWithAt := chunk[0] == 64
				if startsWithAt {

					failedToRespond = true
				}
				isFirstWord = isFirstWord && chunk[0] != 32
				if isFirstWord && startsWithAt {
					chunk = chunk[1:]
				}
			}
			if o.RagReferences && startRefrences {
				refrencesStr += string(chunk)
				return nil
			}
			if /*o.RagReferences &&*/ string(chunk) == "⧉" {
				startRefrences = true
				return nil
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

		messageHistory := []llms.MessageContent{}

		// if memoryStr != "" {
		// 	messageHistory = append(messageHistory, llms.TextParts(llms.ChatMessageTypeSystem, memoryStr))
		// }

		messageHistory = append(messageHistory, llms.TextParts(llms.ChatMessageTypeHuman, Query))
		// 		messageHistory = append(messageHistory, llms.TextParts(llms.ChatMessageTypeSystem, `You are an expert in composing functions. You are given a question and a set of possible functions.
		// Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
		// If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections.

		// If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
		// You SHOULD NOT include any other text in the response.

		// Here is a list of functions in JSON format that you can invoke.

		// `))

		// calloptions = append(calloptions, llms.WithTools(o.Tools.Tools))

		// Token usage calculation should be done here

		resp, err := llmclient.GenerateContent(ctx, messageHistory, llms.WithTools(o.Tools.Tools), llms.WithStreamingFunc(o.StreamingFunc))
		if err != nil {
			return result, err

		}
		respchoice := resp.Choices[0]

		assistantResponse := llms.TextParts(llms.ChatMessageTypeAI, respchoice.Content)
		for _, tc := range respchoice.ToolCalls {
			assistantResponse.Parts = append(assistantResponse.Parts, tc)
		}
		// messageHistory = append(messageHistory, assistantResponse)
		msgs = append(msgs, assistantResponse)

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
							ToolCallID: tc.ID,
							Name:       tc.FunctionCall.Name,
							Content:    fnresult,
						},
					},
				}

				msgs = append(msgs, toolResponse)
			}
		}
		// calloptions = append(calloptions, llms.WithTools(o.Tools.Tools))

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
		result.addAction("Sending Request to LLM", o.ActionCallFunc)
		response, err = llmclient.GenerateContent(ctx,
			msgs,
			calloptions...,
		)
		if err != nil {
			return result, err
		}
	}

	result.addAction("Finished", o.ActionCallFunc)
	memoryAddAllowed = memoryAddAllowed && o.SessionID != ""

	if response != nil {

		// Update memory with the new query if RAG data was found
		if (hasRag || llm.AllowHallucinate || o.AllowHallucinate) && memoryAddAllowed && response.Choices != nil && len(response.Choices) > 0 {
			choiceContent := response.Choices[0].Content
			if o.RagReferences {
				choiceContent = strings.Split(choiceContent, "⧉")[0]
			}
			queryData := MemoryData{
				Question: Query,
				Answer:   choiceContent,
			}

			if !o.PersistentMemory {
				//plain memory
				if exists {
					memoryData = append(memoryData, queryData)
				}
				llm.MemoryManager.AddMemory(o.SessionID, memoryData)

			} else {
				//persistent memory
				tokenUsage, err := llm.PersistentMemoryManager.AddMemory(o.SessionID, queryData)
				if err != nil {
					return result, err
				}
				result.TokenReport.MemorySummarizationTokens = tokenUsage

			}
		}
	}
	if o.PersistentMemory {
		for _, memdoc := range persistentMemoryHistory {
			// page memdoc.PageContent
			memoryData = append(memoryData, extractMemoryData(memdoc.PageContent))
			// memoryData.Keys = append(memoryData.Keys, memdoc.Metadata["keys"])

		}
	}
	result.TokenReport.CompletionTokens.OutputTokens = totalTokens
	result = LLMResult{
		Prompt:          msgs,
		Response:        response,
		RagDocs:         resDocs,
		Memory:          memoryData[:],
		Actions:         result.Actions,
		MemorySummary:   MemorySummary,
		TokenReport:     result.TokenReport,
		FailedToRespond: failedToRespond,
	}
	if o.RagReferences {
		refrencesArray := llmReference{}
		json.Unmarshal([]byte(refrencesStr), &refrencesArray)
		result.LLMReferences = refrencesArray.References
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
