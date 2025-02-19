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
	"strings"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/redisvector"
)

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
