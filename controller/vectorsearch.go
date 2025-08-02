// Copyright (c) 2025 Reza Arani
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
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/redisvector"
)

// HybridSearchResult represents a result from hybrid search with combined scores
type HybridSearchResult struct {
	Document     schema.Document
	VectorScore  float64
	LexicalScore float64
	HybridScore  float64
	SearchType   string
}

// HybridSearchConfig contains configuration for hybrid search
type HybridSearchConfig struct {
	VectorWeight    float64 // Weight for vector similarity (0.0 to 1.0)
	LexicalWeight   float64 // Weight for lexical search (0.0 to 1.0)
	MinVectorScore  float32 // Minimum vector similarity score
	MinLexicalScore float32 // Minimum lexical relevance score
	UseRRF          bool    // Use Reciprocal Rank Fusion instead of weighted scoring
	RRFConstant     float64 // Constant for RRF calculation (default 60)
	MaxResults      int     // Maximum number of results to return
}

// DefaultHybridSearchConfig returns default configuration for hybrid search
func DefaultHybridSearchConfig() HybridSearchConfig {
	return HybridSearchConfig{
		VectorWeight:    0.7,
		LexicalWeight:   0.3,
		MinVectorScore:  0.0,
		MinLexicalScore: 0.0,
		UseRRF:          false,
		RRFConstant:     60.0,
		MaxResults:      50,
	}
}

// CosineSimilarity searches for similar documents in the vector store using cosine similarity.
//
// This function initializes the embedding model, sets up a Redis-based vector store,
// and performs a similarity search based on the provided query.
//
// Parameters:
//   - prefix: A string prefix used to organize and identify related vector entries.
//   - Query: The query string to search for similar documents.
//   - rowCount: The number of results to retrieve from the vector store.
//   - ScoreThreshold: The minimum similarity score threshold for the results.
//
// Returns:
//   - interface{}: The search results containing the most similar documents.
//   - error: An error if the search fails or the embedding model is missing.
func (llm *LLMContainer) CosineSimilarity(prefix, Query string, rowCount int, ScoreThreshold float32) ([]schema.Document, error) {
	var result []schema.Document
	if llm.Embedder == nil {
		return nil, errors.New("missing embedding model")
	} else {
		if !llm.Embedder.initialized() {
			llm.InitEmbedding()
		}
	}

	// Get the embedder from the client
	embedder, err := llm.Embedder.NewEmbedder()
	if err != nil {
		return result, err
	}

	// Setup Redis vector store
	redisVector := redisvector.WithIndexName(prefix+"aillm_vector_idx", true)
	embedderVector := redisvector.WithEmbedder(embedder)

	redisHostURL, redisConnectionErr := llm.getRedisHost()
	if redisConnectionErr != nil {
		return result, redisConnectionErr
	}
	store, err := redisvector.New(context.TODO(), redisvector.WithConnectionURL(redisHostURL), redisVector, embedderVector)
	if err != nil {
		return result, err
	}
	ctx := context.Background()
	optionsVector := []vectorstores.Option{
		vectorstores.WithScoreThreshold(ScoreThreshold),
		vectorstores.WithEmbedder(embedder),
	}
	results, err := store.SimilaritySearch(ctx, Query, rowCount, optionsVector...)
	if err != nil && !strings.Contains(err.Error(), "no such index") {
		return result, fmt.Errorf("search error: %v", err)
	}
	return results, nil
}

// FindKNN performs a K-Nearest Neighbors (KNN) search on the stored vector embeddings.
//
// This function retrieves the most relevant documents based on the provided query,
// using the KNN algorithm to rank them according to their proximity in the vector space.
//
// Parameters:
//   - prefix: A string prefix used to identify relevant vector entries.
//   - searchQuery: The query string to find the nearest neighbors for.
//   - rowCount: The number of closest neighbors to retrieve.
//   - ScoreThreshold: The minimum similarity score for considering results.
//
// Returns:
//   - []schema.Document: The retrieved relevant documents.
//   - error: An error if the search fails or the embedding model is missing.
func (llm *LLMContainer) FindKNN(prefix, searchQuery string, rowCount int, ScoreThreshold float32) ([]schema.Document, error) {
	var result []schema.Document

	// llm.CosineSimilarity(prefix, searchQuery,rowCount,ScoreThreshold)
	embedder, err := llm.Embedder.NewEmbedder()
	if err != nil {
		return result, err
	}

	redisVector := redisvector.WithIndexName(prefix+"aillm_vector_idx", true)
	embedderVector := redisvector.WithEmbedder(embedder)
	redisHostURL, redisConnectionErr := llm.getRedisHost()
	if redisConnectionErr != nil {
		return result, redisConnectionErr
	}

	store, err := redisvector.New(context.TODO(), redisvector.WithConnectionURL(redisHostURL), redisVector, embedderVector)
	if err != nil {
		return result, err
	}

	optionsVector := []vectorstores.Option{
		vectorstores.WithScoreThreshold(ScoreThreshold),
		vectorstores.WithEmbedder(embedder),
	}

	retriever := vectorstores.ToRetriever(store, rowCount, optionsVector...)

	resDocs, err := retriever.GetRelevantDocuments(context.Background(), searchQuery)
	if err != nil {
		return result, err
	}
	return resDocs, nil
}

// HybridSearch performs a hybrid search combining vector similarity and lexical search for improved accuracy.
//
// This function combines the power of semantic vector search with traditional keyword-based search
// to provide more accurate and comprehensive results. It uses techniques like Reciprocal Rank Fusion (RRF)
// to merge results from both search methods.
//
// Parameters:
//   - prefix: A string prefix used to identify relevant vector entries.
//   - searchQuery: The query string to search for.
//   - rowCount: The number of results to retrieve.
//   - ScoreThreshold: The minimum similarity score threshold for results.
//   - config: Configuration for hybrid search behavior.
//
// Returns:
//   - []schema.Document: The retrieved relevant documents with hybrid scores.
//   - error: An error if the search fails or required components are missing.
func (llm *LLMContainer) HybridSearch(prefix, searchQuery string, rowCount int, ScoreThreshold float32, config *HybridSearchConfig) ([]schema.Document, error) {
	if config == nil {
		defaultConfig := DefaultHybridSearchConfig()
		config = &defaultConfig
	}

	// Validate configuration
	if config.VectorWeight < 0 || config.VectorWeight > 1 {
		return nil, errors.New("vector weight must be between 0 and 1")
	}
	if config.LexicalWeight < 0 || config.LexicalWeight > 1 {
		return nil, errors.New("lexical weight must be between 0 and 1")
	}
	if config.VectorWeight+config.LexicalWeight != 1.0 {
		// Normalize weights
		total := config.VectorWeight + config.LexicalWeight
		config.VectorWeight = config.VectorWeight / total
		config.LexicalWeight = config.LexicalWeight / total
	}

	// Perform vector similarity search
	vectorResults, err := llm.performVectorSearch(prefix, searchQuery, config.MaxResults, config.MinVectorScore)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %v", err)
	}

	// Perform lexical search
	lexicalResults, err := llm.performLexicalSearch(prefix, searchQuery, config.MaxResults, config.MinLexicalScore)
	if err != nil {
		return nil, fmt.Errorf("lexical search failed: %v", err)
	}

	// Combine results using hybrid scoring
	hybridResults := llm.combineSearchResults(vectorResults, lexicalResults, config)

	// Sort by hybrid score (descending - higher scores are better)
	sort.Slice(hybridResults, func(i, j int) bool {
		return hybridResults[i].HybridScore > hybridResults[j].HybridScore
	})

	// Convert to schema.Document slice and limit results
	var finalResults []schema.Document
	limit := rowCount
	if limit > len(hybridResults) {
		limit = len(hybridResults)
	}

	for i := 0; i < limit; i++ {
		doc := hybridResults[i].Document
		// Add hybrid score to metadata
		if doc.Metadata == nil {
			doc.Metadata = make(map[string]interface{})
		}
		doc.Metadata["hybrid_score"] = hybridResults[i].HybridScore
		doc.Metadata["vector_score"] = hybridResults[i].VectorScore
		doc.Metadata["lexical_score"] = hybridResults[i].LexicalScore
		doc.Metadata["search_type"] = hybridResults[i].SearchType
		doc.Score = float32(hybridResults[i].HybridScore)
		finalResults = append(finalResults, doc)
	}

	return finalResults, nil
}

// performVectorSearch executes vector similarity search
func (llm *LLMContainer) performVectorSearch(prefix, searchQuery string, maxResults int, minScore float32) ([]HybridSearchResult, error) {
	if llm.Embedder == nil {
		return nil, errors.New("missing embedding model")
	}

	if !llm.Embedder.initialized() {
		llm.InitEmbedding()
	}

	// Get the embedder from the client
	embedder, err := llm.Embedder.NewEmbedder()
	if err != nil {
		return nil, err
	}

	// Setup Redis vector store
	redisVector := redisvector.WithIndexName(prefix+"aillm_vector_idx", true)
	embedderVector := redisvector.WithEmbedder(embedder)

	redisHostURL, redisConnectionErr := llm.getRedisHost()
	if redisConnectionErr != nil {
		return nil, redisConnectionErr
	}

	store, err := redisvector.New(context.TODO(), redisvector.WithConnectionURL(redisHostURL), redisVector, embedderVector)
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	optionsVector := []vectorstores.Option{
		vectorstores.WithScoreThreshold(minScore),
		vectorstores.WithEmbedder(embedder),
	}

	results, err := store.SimilaritySearch(ctx, searchQuery, maxResults, optionsVector...)
	if err != nil && !strings.Contains(err.Error(), "no such index") {
		return nil, fmt.Errorf("vector search error: %v", err)
	}

	var hybridResults []HybridSearchResult
	for _, doc := range results {
		hybridResults = append(hybridResults, HybridSearchResult{
			Document:     doc,
			VectorScore:  float64(doc.Score),
			LexicalScore: 0.0,
			HybridScore:  0.0,
			SearchType:   "vector",
		})
	}

	return hybridResults, nil
}

// performLexicalSearch executes lexical/keyword search using Redis FT.SEARCH
func (llm *LLMContainer) performLexicalSearch(prefix, searchQuery string, maxResults int, minScore float32) ([]HybridSearchResult, error) {
	rdb := llm.RedisClient.redisClient
	ctx := context.Background()

	// Create a text index name for lexical search
	textIndexName := prefix + "aillm_text_idx"

	// Ensure text index exists
	err := llm.createTextIndex(textIndexName, prefix)
	if err != nil {
		return nil, fmt.Errorf("failed to create text index: %v", err)
	}

	// Perform FT.SEARCH query for lexical search
	// Search in both content and title fields
	keywords := []string{}
	lines := strings.Split(searchQuery, "\n")
	for _, line := range lines {
		// Split the text into words using whitespace and common delimiters
		re := regexp.MustCompile(`[ ,\.]+`)
		words := re.Split(line, -1)
		for _, word := range words {
			if len(word) > 2 {
				keywords = append(keywords, llm.escapeRedisSearchQuery(word))
			}
		}
	}

	// Build the final search query using OR logic
	finalSearchQuery := ""
	for i, keyword := range keywords {
		if i > 0 {
			finalSearchQuery += " | "
		}
		finalSearchQuery += fmt.Sprintf("(@content:*%s*)", keyword)
	}

	// If no valid keywords found, return empty results
	if finalSearchQuery == "" {
		return []HybridSearchResult{}, nil
	}

	// if strings.Contains(searchQuery, "\n") {
	// 	// convert each line to a separate redis "OR" query
	// 	lines := strings.Split(searchQuery, "\n")
	// 	query := ""
	// 	for idx, line := range lines {
	// 		searchExpression := fmt.Sprintf("(@content:*%s*)", llm.escapeRedisSearchQuery(line))
	// 		if idx > 0 {
	// 			query += " | "
	// 		}
	// 		query += searchExpression
	// 	}
	// 	searchQuery = query
	// } else {
	// 	searchQuery = fmt.Sprintf("(@content:*%s*)", llm.escapeRedisSearchQuery(searchQuery))
	// }

	searchResults, err := rdb.Do(ctx,
		"FT.SEARCH", textIndexName,
		finalSearchQuery,
		"LIMIT", 0, maxResults,
		"WITHSCORES").Result()

	if err != nil {
		return nil, fmt.Errorf("lexical search error: %v", err)
	}

	// Parse Redis FT.SEARCH results
	return llm.parseRedisSearchResults(searchResults, "lexical")
}

// createTextIndex creates a text index for lexical search if it doesn't exist
func (llm *LLMContainer) createTextIndex(indexName, prefix string) error {
	rdb := llm.RedisClient.redisClient
	ctx := context.Background()

	// Check if index exists
	_, err := rdb.Do(ctx, "FT.INFO", indexName).Result()
	if err == nil {
		return nil // Index already exists
	}

	// Create text index for lexical search
	_, err = rdb.Do(ctx,
		"FT.CREATE", indexName,
		"ON", "HASH",
		"PREFIX", "1", "doc:"+prefix,
		"SCHEMA",
		"content", "TEXT").Result()

	return err
}

// escapeRedisSearchQuery escapes special characters for Redis FT.SEARCH
func (llm *LLMContainer) escapeRedisSearchQuery(query string) string {
	// Replace special characters that have meaning in Redis FT.SEARCH
	// We need to escape these characters to prevent syntax errors
	replacements := map[string]string{
		"\\": "\\\\", // Backslash must be escaped first
		"@":  "\\@",
		"(":  "\\(",
		")":  "\\)",
		"[":  "\\[",
		"]":  "\\]",
		"{":  "\\{",
		"}":  "\\}",
		"*":  "\\*",
		"+":  "\\+",
		"?":  "\\?",
		"|":  "\\|",
		"^":  "\\^",
		"$":  "\\$",
		"-":  "\\-",
		"=":  "\\=",
		"~":  "\\~",
		":":  "\\:",
		";":  "\\;",
		"!":  "\\!",
		"#":  "\\#",
		"%":  "\\%",
		"&":  "\\&",
		"'":  "\\'",
		"\"": "\\\"",
		"/":  "\\/", // Forward slash should also be escaped
	}

	escaped := query
	// Process backslash first to avoid double escaping
	escaped = strings.ReplaceAll(escaped, "\\", "\\\\")

	// Then process all other special characters
	for old, new := range replacements {
		if old != "\\" { // Skip backslash as it's already processed
			escaped = strings.ReplaceAll(escaped, old, new)
		}
	}

	return escaped
}

// parseRedisSearchResults parses Redis FT.SEARCH results into HybridSearchResult format
func (llm *LLMContainer) parseRedisSearchResults(results interface{}, searchType string) ([]HybridSearchResult, error) {
	var hybridResults []HybridSearchResult

	// Handle the new Redis response format
	resultMap, ok := results.(map[interface{}]interface{})
	if !ok {
		return hybridResults, nil
	}

	// Extract the results array from the new format
	resultsArray, ok := resultMap["results"]
	if !ok {
		return hybridResults, nil
	}

	resultsSlice, ok := resultsArray.([]interface{})
	if !ok {
		return hybridResults, nil
	}

	// Parse each result in the results array
	for _, resultItem := range resultsSlice {
		// Each result item should be a map with "id", "score", and "values" fields
		resultMap, ok := resultItem.(map[interface{}]interface{})
		if !ok {
			continue
		}

		// Extract document ID
		docKey, ok := resultMap["id"].(string)
		if !ok {
			continue
		}

		// Extract score
		var score float64
		if scoreVal, ok := resultMap["score"]; ok {
			switch v := scoreVal.(type) {
			case string:
				score = llm.parseFloat(v)
			case float64:
				score = v
			case int64:
				score = float64(v)
			}
		}

		// Extract document values/fields
		values, ok := resultMap["extra_attributes"].(map[interface{}]interface{})
		if !ok {
			continue
		}

		// Extract document content
		doc := schema.Document{
			Metadata: make(map[string]interface{}),
		}
		doc.Metadata["id"] = docKey

		// Parse field-value pairs from the values map
		for fieldName, fieldValue := range values {
			fieldNameStr, ok := fieldName.(string)
			if !ok {
				continue
			}

			fieldValueStr, ok := fieldValue.(string)
			if !ok {
				continue
			}

			switch fieldNameStr {
			case "content":
				doc.PageContent = fieldValueStr
			case "rawkey":
				doc.Metadata["rawkey"] = fieldValueStr
			case "Keywords":
				doc.Metadata["keywords"] = fieldValueStr
			case "sources":
				doc.Metadata["sources"] = fieldValueStr
			}
		}

		hybridResults = append(hybridResults, HybridSearchResult{
			Document:     doc,
			VectorScore:  0.0,
			LexicalScore: score,
			HybridScore:  0.0,
			SearchType:   searchType,
		})
	}

	return hybridResults, nil
}

// parseFloat safely parses a string to float64
func (llm *LLMContainer) parseFloat(s string) float64 {
	if val, err := strconv.ParseFloat(s, 64); err == nil {
		return val
	}
	return 0.0
}

// combineSearchResults combines vector and lexical search results using hybrid scoring
func (llm *LLMContainer) combineSearchResults(vectorResults, lexicalResults []HybridSearchResult, config *HybridSearchConfig) []HybridSearchResult {
	// Create a map to store combined results by document ID
	resultMap := make(map[string]HybridSearchResult)

	// Add vector results
	for i, result := range vectorResults {
		docID := llm.getDocumentID(result.Document)
		if config.UseRRF {
			result.HybridScore = llm.calculateRRF(i+1, 0, config.RRFConstant, config.VectorWeight, config.LexicalWeight)
		} else {
			result.HybridScore = config.VectorWeight * result.VectorScore
		}
		result.SearchType = "vector"
		resultMap[docID] = result
	}

	// Add or merge lexical results
	for i, result := range lexicalResults {
		docID := llm.getDocumentID(result.Document)
		if existing, exists := resultMap[docID]; exists {
			// Document found in both searches - merge scores
			if config.UseRRF {
				vectorRank := llm.findRank(docID, vectorResults)
				lexicalRank := i + 1
				existing.HybridScore = llm.calculateRRF(vectorRank, lexicalRank, config.RRFConstant, config.VectorWeight, config.LexicalWeight)
			} else {
				existing.HybridScore = config.VectorWeight*existing.VectorScore + config.LexicalWeight*result.LexicalScore
			}
			existing.LexicalScore = result.LexicalScore
			existing.SearchType = "hybrid"
			resultMap[docID] = existing
		} else {
			// Document only found in lexical search
			if config.UseRRF {
				result.HybridScore = llm.calculateRRF(0, i+1, config.RRFConstant, config.VectorWeight, config.LexicalWeight)
			} else {
				result.HybridScore = config.LexicalWeight * result.LexicalScore
			}
			result.SearchType = "lexical"
			resultMap[docID] = result
		}
	}

	// Convert map to slice
	var finalResults []HybridSearchResult
	for _, result := range resultMap {
		finalResults = append(finalResults, result)
	}

	return finalResults
}

// calculateRRF calculates the Reciprocal Rank Fusion score
func (llm *LLMContainer) calculateRRF(vectorRank, lexicalRank int, constant, vectorWeight, lexicalWeight float64) float64 {
	score := 0.0

	if vectorRank > 0 {
		score += vectorWeight * (1.0 / (constant + float64(vectorRank)))
	}

	if lexicalRank > 0 {
		score += lexicalWeight * (1.0 / (constant + float64(lexicalRank)))
	}

	return score
}

// getDocumentID extracts a unique identifier from a document
func (llm *LLMContainer) getDocumentID(doc schema.Document) string {
	if id, exists := doc.Metadata["id"]; exists {
		return fmt.Sprintf("%v", id)
	}

	// Fallback to content hash if no ID
	return fmt.Sprintf("%x", hash(doc.PageContent))
}

// findRank finds the rank of a document in a result set
func (llm *LLMContainer) findRank(docID string, results []HybridSearchResult) int {
	for i, result := range results {
		if llm.getDocumentID(result.Document) == docID {
			return i + 1
		}
	}
	return 0
}

// hash generates a simple hash for a string
func hash(s string) uint32 {
	h := uint32(0)
	for _, c := range s {
		h = h*31 + uint32(c)
	}
	return h
}

// performLexicalSearchOnly performs lexical search only and returns results in standard format
//
// This function performs only lexical/keyword search without vector similarity.
//
// Parameters:
//   - prefix: A string prefix used to identify relevant vector entries.
//   - searchQuery: The query string to search for.
//   - rowCount: The number of results to retrieve.
//   - ScoreThreshold: The minimum similarity score threshold for results.
//
// Returns:
//   - []schema.Document: The retrieved relevant documents.
//   - error: An error if the search fails.
func (llm *LLMContainer) performLexicalSearchOnly(prefix, searchQuery string, rowCount int, ScoreThreshold float32) ([]schema.Document, error) {
	// Perform lexical search
	hybridResults, err := llm.performLexicalSearch(prefix, searchQuery, rowCount, ScoreThreshold)
	if err != nil {
		return nil, fmt.Errorf("lexical search failed: %v", err)
	}

	// Sort by lexical score (descending)
	sort.Slice(hybridResults, func(i, j int) bool {
		return hybridResults[i].LexicalScore > hybridResults[j].LexicalScore
	})

	// Convert to schema.Document slice
	var finalResults []schema.Document
	limit := rowCount
	if limit > len(hybridResults) {
		limit = len(hybridResults)
	}

	for i := 0; i < limit; i++ {
		doc := hybridResults[i].Document
		// Add lexical score to metadata
		if doc.Metadata == nil {
			doc.Metadata = make(map[string]interface{})
		}
		doc.Metadata["lexical_score"] = hybridResults[i].LexicalScore
		doc.Metadata["search_type"] = "lexical"
		doc.Score = float32(hybridResults[i].LexicalScore)
		finalResults = append(finalResults, doc)
	}

	return finalResults, nil
}

// SemanticSearch performs enhanced semantic search using the best available algorithm
//
// This function automatically selects the most appropriate search algorithm based on
// the system configuration and query characteristics.
//
// Parameters:
//   - prefix: A string prefix used to identify relevant vector entries.
//   - searchQuery: The query string to search for.
//   - rowCount: The number of results to retrieve.
//   - ScoreThreshold: The minimum similarity score threshold for results.
//
// Returns:
//   - []schema.Document: The retrieved relevant documents.
//   - error: An error if the search fails.
func (llm *LLMContainer) SemanticSearch(prefix, searchQuery string, rowCount int, ScoreThreshold float32) ([]schema.Document, error) {
	// Use hybrid search for better accuracy
	config := DefaultHybridSearchConfig()
	config.MaxResults = rowCount * 2 // Get more results for better fusion

	return llm.HybridSearch(prefix, searchQuery, rowCount, ScoreThreshold, &config)
}
