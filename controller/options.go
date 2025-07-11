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

import "context"

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
		// if Prefix == "" {
		// 	Prefix = "default"
		// }
		o.Prefix = Prefix
	}
}

func (llm *LLMContainer) WithEmbeddingIndex(Index string) LLMCallOption {
	return func(o *LLMCallOptions) {
		// if Index == "" {
		// 	Index = "default"
		// }
		o.Index = Index
	}
}

// SearchAll specifies the scope of search,
//
// Parameters:
//   - language: scope of general search in specific language
//
// Returns:
//   - LLMCallOption: An option that sets the embedding prefix.
func (llm *LLMContainer) SearchAll(language string) LLMCallOption {
	return func(o *LLMCallOptions) {
		// o.Prefix = "all:"+language
		o.Language = language
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
	// if o.Prefix == "" {
	// 	o.Prefix = "default"
	// }
	return o.Prefix
}

// WithEmbeddingPrefix specifies a prefix for identifying related embeddings.
//
// Parameters:
//   - Prefix: A string prefix used to group or identify embeddings in the store.
//
// Returns:
//   - LLMCallOption: An option that sets the embedding prefix.
func (llm *LLMContainer) WithLimitGeneralEmbedding(denied bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.LimitGeneralEmbedding = denied
	}
}

// WithCotextCleanup Cleanup retrieved context, specially html codes to make a clear context
//
// Parameters:
//   - cleanup: A boolean value to update property
//
// Returns:
//   - LLMCallOption: An option that sets the embedding prefix.
func (llm *LLMContainer) WithCotextCleanup(cleanup bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.CotextCleanup = cleanup
	}
}

// WithPersistentMemory enhances the memory by using vector search to create more efficient prompts for conversation memory
//
// Parameters:
//   - usePersistentMemory: A boolean value to update property
//
// Returns:
//   - LLMCallOption: An option that sets the embedding prefix.
func (llm *LLMContainer) WithPersistentMemory(usePersistentMemory bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.PersistentMemory = usePersistentMemory
	}
}

// WithPersistentMemory enhances the memory by using vector search to create more efficient prompts for conversation memory
//
// Parameters:
//   - usePersistentMemory: A boolean value to update property
//
// Returns:
//   - LLMCallOption: An option that sets the embedding prefix.
func (llm *LLMContainer) WithCharacter(character string) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.character = character
	}
}

// WithMaxTokens specifies the maximum tokens of response.
//
// Parameters:
//   - maxTokens: integer value of Maximum allowed tokens
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithMaxTokens(maxTokens int) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.MaxTokens = maxTokens
	}
}

// WithLanguageChannel returns user language and send it to main thread
//
// Parameters:
//   - userChannel: AILLM will send language to selected channel for post processing.
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithLanguageChannel(userChannel chan string) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.LanguageChannel = userChannel
	}
}

// WithForcedLanguage Forces ExactPrompt output language
//
// Parameters:
//   - Language: The language in which the LLM should generate responses.
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithForcedLanguage(forceUserLanguage bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.ForceLanguage = forceUserLanguage
	}
}

// WithAllowHallucinate allows model to hallucinate
//
// Parameters:
//   - AllowHallucinate: A boolean value to update property
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithAllowHallucinate(AllowHallucinate bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.AllowHallucinate = AllowHallucinate
	}
}

// WithForceLLMToAnswerLong forces LLM to answer long
//
// Parameters:
//   - forceLong: A boolean value to update property
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithForceLLMToAnswerLong(forceLong bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.ForceLLMToAnswerLong = forceLong
	}
}

// WithLLMSpliter uses LLM to split text
//
// Parameters:
//   - UseLLMToSplitText: A boolean value to update property
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithLLMSpliter(UseLLMToSplitText bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.UseLLMToSplitText = UseLLMToSplitText
	}
}

// WithIncludeDate include date in prompt
//
// Parameters:
//   - IncludeDate: A boolean value to update property
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithIncludeDate(IncludeDate bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.IncludeDate = IncludeDate
	}
}

// WithIncludeDate include date in prompt
//
// Parameters:
//   - IncludeDate: A boolean value to update property
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithRagReferences(RagReferences bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.RagReferences = RagReferences
	}
}

// WithTools specifies the tools to use for the query.
//
// Parameters:
//   - tools: A list of tools to use for the query.
//
// Returns:
//   - LLMCallOption: An option that sets the tools.
//
// Experimental - Just works with OpenAI
func (llm *LLMContainer) WithTools(tools AillmTools) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.Tools = tools
	}
}

// WithSearchAlgorithm specifies the search algorithm to use for the query.
//
// Parameters:
//   - SearchAlgorithm: The search algorithm to use for the query.
//
// Returns:
//   - LLMCallOption: An option that sets the search algorithm.
func (llm *LLMContainer) WithSearchAlgorithm(SearchAlgorithm int) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.SearchAlgorithm = SearchAlgorithm
	}
}

// WithIgnoreSecurityCheck ignores security check
//
// Parameters:
//   - IgnoreSecurityCheck: A boolean value to update property
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithIgnoreSecurityCheck(ignoreSecurityCheck bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.ignoreSecurityCheck = ignoreSecurityCheck
	}
}

// WithHybridSearch enables hybrid search combining vector similarity and lexical search
func (llm *LLMContainer) WithHybridSearch() LLMCallOption {
	return func(o *LLMCallOptions) {
		o.SearchAlgorithm = HybridSearch
	}
}

// WithLexicalSearch enables lexical/keyword search only
func (llm *LLMContainer) WithLexicalSearch() LLMCallOption {
	return func(o *LLMCallOptions) {
		o.SearchAlgorithm = LexicalSearch
	}
}

// WithSemanticSearch enables enhanced semantic search (auto-selects best algorithm)
func (llm *LLMContainer) WithSemanticSearch() LLMCallOption {
	return func(o *LLMCallOptions) {
		o.SearchAlgorithm = SemanticSearch
	}
}

// WithSimilaritySearch enables cosine similarity search (default)
func (llm *LLMContainer) WithSimilaritySearch() LLMCallOption {
	return func(o *LLMCallOptions) {
		o.SearchAlgorithm = SimilaritySearch
	}
}

// WithKNNSearch enables K-Nearest Neighbors search
func (llm *LLMContainer) WithKNNSearch() LLMCallOption {
	return func(o *LLMCallOptions) {
		o.SearchAlgorithm = KNearestNeighbors
	}
}

// WithDebug enables debug mode
func (llm *LLMContainer) WithDebug(debug bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.debug = debug
	}
}

// WithMaxWords specifies the maximum words of response.
//
// Parameters:
//   - maxTokens: integer value of Maximum allowed tokens
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithMaxWords(maxWords int) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.maxWords = maxWords
	}
}

// WithCustomModel specifies the custom mode of response.
//
// Parameters:
//   - maxTokens: integer value of Maximum allowed tokens
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithCustomModel(customModel string) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.customModel = customModel
	}
}

// WithAsyncMemorySummarization specifies the async memory summarization.
//
// Parameters:
//   - asyncMemorySummarization: boolean value of async memory summarization
//
// Returns:
//   - LLMCallOption: An option that sets the query language.
func (llm *LLMContainer) WithAsyncMemorySummarization(asyncMemorySummarization bool) LLMCallOption {
	return func(o *LLMCallOptions) {
		o.asyncMemorySummarization = asyncMemorySummarization
	}
}
